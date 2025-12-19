"""Template for Ray-based applications"""

import torch
from pathlib import Path
from ray import serve

from torch_split.runtime import SwitchboardRuntime
from opentelemetry import trace, context
from opentelemetry.propagate import extract


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0})
class ComponentTemplate:
    def __init__(self):
        ### BEGIN TEMPLATE
        switchboard_path = Path("path/to/switchboard")
        switchboard_label = "component_label"
        output_map = {1: "text_embeds_1"}
        ### END TEMPLATE

        if switchboard_path.exists():
            self.switchboard = SwitchboardRuntime((switchboard_path), load_only=[switchboard_label])
            self.label = switchboard_label
            self.output_map = output_map
            self.tracer = trace.get_tracer(self.__class__.__name__)
        else:
            raise RuntimeError(f"Switchboard path {switchboard_path} does not exist.")

    def single(self, args: dict) -> dict:
        with self.tracer.start_as_current_span(f"{self.__class__.__name__}::single_call") as span:
            # Move inputs to GPU
            for k in args.keys():
                args[k] = torch.tensor(args[k], device="cuda")
            span.add_event("inputs_moved_to_gpu")

            # Call the switchboard
            r = self.switchboard.call(self.label, **args)
            span.add_event("switchboard_call_completed")

            # Move outputs to CPU and map to output names
            output = {}
            for out_idx, out_name in self.output_map.items():
                output[out_name] = r[out_idx].detach().cpu().numpy()
            span.add_event("outputs_moved_to_cpu")

            return output

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)  # type: ignore
    async def __call__(self, batch: list[dict], _ts_trace_ctx: list[dict]) -> list[dict]:
        with self.tracer.start_as_current_span(f"{self.__class__.__name__}::__call__") as span:
            span.set_attribute("batch_size", len(batch))

            # re-hydrate individual trace contexts
            spans = []
            ctxs = []

            for carrier in _ts_trace_ctx:
                parent_ctx = extract(carrier)
                span = self.tracer.start_span(f"{self.__class__.__name__}::__call__::request", context=parent_ctx)
                spans.append(span)
                ctxs.append(trace.set_span_in_context(span))
            span.add_event("trace_contexts_rehydrated")

            # process the batch
            results = []
            for item, span_ctx in zip(batch, ctxs):
                token = context.attach(span_ctx)
                try:
                    results.append(self.single(item))
                finally:
                    context.detach(token)

            # clean up individual spans
            span.add_event("batch_processing_completed")
            for s in spans:
                s.end()
            span.add_event("individual_spans_ended")

        return results
