import asyncio
import time
from typing import List

from fastapi import FastAPI
from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

from examples.ray_test.hook import setup_tracing

tracer = trace.get_tracer("myapp")
prop = propagate.get_global_textmap()

app = FastAPI()


@serve.deployment(max_ongoing_requests=64)
@serve.ingress(app)
class Ingress:
    def __init__(self):  # , model: DeploymentHandle):
        setup_tracing()

    @app.post("/infer")
    async def pipeline(self, request: Request):
        # Extract incoming trace context from HTTP headers (traceparent, etc.)
        parent_ctx = propagate.extract(dict(request.headers))

        body = await request.json()
        print("here")

        with tracer.start_as_current_span("ingress", context=parent_ctx, kind=SpanKind.SERVER):
            with tracer.start_as_current_span("call.batched_model", kind=SpanKind.CLIENT):
                carrier: dict[str, str] = {}
                propagate.inject(carrier)

                item = BatchedItem(
                    payload=body,
                    carrier=carrier,
                    enqueue_time_ns=time.time_ns(),
                )

                # DeploymentHandle returns a DeploymentResponse that you can await inside a deployment :contentReference[oaicite:1]{index=1}
                result = await self.model.remote(item)
                return JSONResponse(result)


app = Ingress.bind()


# @serve.deployment
# class MyBatchedStage:
#     async def _forward(self, batch_payloads: List[Any]) -> List[Any]:
#         await asyncio.sleep(0.05)  # Simulate processing delay
#         return [payload for payload in batch_payloads]

#     @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
#     async def __call__(self, items: List[BatchedItem]) -> List[Any]:
#         # Ray Serve provides per-item request contexts for the batch (aligned with items)
#         serve_req_ctxs = serve.context._get_serve_batch_request_context()  # internal API
#         batch_start_ns = time.time_ns()

#         # 1) Per-request queue wait spans (show up in each request trace)
#         request_span_contexts = []
#         for i, item in enumerate(items):
#             parent_ctx = prop.extract(item.otel_carrier)
#             parent_span_ctx = trace.get_current_span(parent_ctx).get_span_context()
#             request_span_contexts.append(parent_span_ctx)

#             wait = tracer.start_span(
#                 "rayserve.batch.wait",
#                 context=parent_ctx,
#                 kind=SpanKind.INTERNAL,
#                 start_time=item.enqueue_time_ns,
#                 attributes={
#                     "ray.serve.batch.index": i,
#                     "ray.serve.batch.size": len(items),
#                 },
#             )

#             # Optional: correlate with Ray Serve request_id/route if available
#             if i < len(serve_req_ctxs):
#                 wait.set_attribute("ray.serve.request_id", serve_req_ctxs[i].request_id)
#                 wait.set_attribute("ray.serve.route", serve_req_ctxs[i].route)

#             wait.end(end_time=batch_start_ns)

#         # 2) Shared batch execution span, linked to all requests (fan-in)
#         # Links should ideally be provided at span creation (helps samplers). :contentReference[oaicite:5]{index=5}
#         links = [Link(ctx, attributes={"ray.serve.batch.index": i}) for i, ctx in enumerate(request_span_contexts)]

#         batch_span = tracer.start_span(
#             "rayserve.batch.execute",
#             kind=SpanKind.INTERNAL,
#             links=links,
#             start_time=batch_start_ns,
#             attributes={"ray.serve.batch.size": len(items)},
#         )

#         try:
#             # Make batch_span current so subcomponent spans nest under it
#             with trace.use_span(batch_span, end_on_exit=False):
#                 with tracer.start_as_current_span("model.forward"):
#                     results = await self._forward([x.payload for x in items])

#                 with tracer.start_as_current_span("postprocess"):
#                     results = self._postprocess(results)

#             return results
#         except Exception as e:
#             batch_span.record_exception(e)
#             raise
#         finally:
#             batch_span.end()
