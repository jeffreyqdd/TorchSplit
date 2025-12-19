# client.py
import asyncio
import time
import httpx

from opentelemetry import propagate, trace

# OTel SDK (demo: prints spans to stdout)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

resource = Resource.create({"service.name": "ray-serve-client-demo"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ray-serve-client-demo")

URL = "http://127.0.0.1:8000/infer"


async def one_call(i: int, client: httpx.AsyncClient):
    headers: dict[str, str] = {}

    # Create a client span per request and inject trace context into HTTP headers
    with tracer.start_as_current_span("client.infer") as span:
        propagate.inject(headers)
        t0 = time.time()

        r = await client.post(URL, json={"x": i}, headers=headers)
        span.set_attribute("http.status_code", r.status_code)
        span.set_attribute("client.latency_ms", (time.time() - t0) * 1000.0)
        return r.json()


async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Fire many concurrent requests to trigger opportunistic batching
        results = await asyncio.gather(*[one_call(i, client) for i in range(100)])

    print("First 5 results:", results[:5])


if __name__ == "__main__":
    asyncio.run(main())
