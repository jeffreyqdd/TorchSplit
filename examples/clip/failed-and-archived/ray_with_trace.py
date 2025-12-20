import ray
from ray import serve

from opentelemetry import trace, context as otel_context
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def traced_serve_call(
    span_name: str,
    *,
    batched: bool = False,
):
    """
    Decorator for Ray Serve __call__ methods that:
    - extracts W3C trace context from kwargs
    - creates child span (non-batched)
    - creates ONE span with LINKS (batched)
    """

    def decorator(fn):
        async def wrapper(self, *args, **kwargs):
            tracer = getattr(self, "tracer", trace.get_tracer(__name__))

            # -------------------------
            # Batched Serve deployment
            # -------------------------
            if batched:
                batch = args[0]  # Serve passes list[(args, kwargs)]

                links = []
                clean_batch = []

                for call_args, call_kwargs in batch:
                    carrier = call_kwargs.pop("_trace_ctx", None)
                    if carrier:
                        ctx = extract(carrier)
                        span_ctx = trace.get_current_span(ctx).get_span_context()
                        links.append(Link(span_ctx))
                    clean_batch.append((call_args, call_kwargs))

                with tracer.start_as_current_span(
                    span_name,
                    links=links,
                ):
                    return await fn(self, clean_batch)

            # -------------------------
            # Non-batched deployment
            # -------------------------
            else:
                carrier = kwargs.pop("_trace_ctx", None)
                token = None

                if carrier:
                    ctx = extract(carrier)
                    token = otel_context.attach(ctx)

                try:
                    with tracer.start_as_current_span(span_name):
                        return await fn(self, *args, **kwargs)
                finally:
                    if token is not None:
                        otel_context.detach(token)

        return wrapper

    return decorator


# ----------------------------
# Tracing init (safe, once per process)
# ----------------------------
def init_tracing():
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        return provider

    resource = Resource.create({"service.name": "ray-test"})

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


# ----------------------------
# Deployment A
# ----------------------------
@serve.deployment
class A:
    def __init__(self):
        init_tracing()
        self.tracer = trace.get_tracer("A")

    async def __call__(self, inputs, _trace_ctx=None):
        token = None
        if _trace_ctx is not None:
            ctx = extract(_trace_ctx)
            token = otel_context.attach(ctx)

        try:
            with self.tracer.start_as_current_span("A::encode"):
                return {"to_5": inputs["l_pixel_values_"] * 5}
        finally:
            if token is not None:
                otel_context.detach(token)


# ----------------------------
# Deployment B
# ----------------------------
@serve.deployment
class B:
    def __init__(self):
        init_tracing()
        self.tracer = trace.get_tracer("B")

    async def __call__(self, inputs, _trace_ctx=None):
        token = None
        if _trace_ctx is not None:
            ctx = extract(_trace_ctx)
            token = otel_context.attach(ctx)

        try:
            with self.tracer.start_as_current_span("B::embed"):
                return {"text_embeds_1": inputs["l_input_ids_"] + 1}
        finally:
            if token is not None:
                otel_context.detach(token)


# ----------------------------
# Deployment C
# ----------------------------
@serve.deployment
class C:
    def __init__(self):
        init_tracing()
        self.tracer = trace.get_tracer("C")

    async def __call__(self, inputs, _trace_ctx=None):
        token = None
        if _trace_ctx is not None:
            ctx = extract(_trace_ctx)
            token = otel_context.attach(ctx)

        try:
            with self.tracer.start_as_current_span("C::merge"):
                return inputs["to_5"] + inputs["text_embeds_1"]
        finally:
            if token is not None:
                otel_context.detach(token)


# ----------------------------
# Pipeline (parent span)
# ----------------------------
@serve.deployment
class Pipeline:
    def __init__(self, A_handle, B_handle, C_handle):
        init_tracing()
        self.tracer = trace.get_tracer("pipeline")
        self.A = A_handle
        self.B = B_handle
        self.C = C_handle

    async def __call__(self, request):
        with self.tracer.start_as_current_span("pipeline_job"):
            payload = await request.json()

            ctx = {}
            inject(ctx, otel_context.get_current())
            A_ref = self.A.remote({"l_pixel_values_": payload["pixel_values"]}, _trace_ctx=ctx)

            B_ref = self.B.remote(
                {
                    "l_input_ids_": payload["input_ids"],
                    "l_attention_mask_": payload.get("attention_mask"),
                },
                _trace_ctx=ctx,
            )

            A_out, B_out = await A_ref, await B_ref

            C_out = await self.C.remote(
                {
                    "to_5": A_out["to_5"],
                    "text_embeds_1": B_out["text_embeds_1"],
                },
                _trace_ctx=ctx,
            )

            return {"result": C_out}


app = Pipeline.bind(
    A.bind(),
    B.bind(),
    C.bind(),
)
