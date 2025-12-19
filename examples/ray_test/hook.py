from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Link, SpanKind


TRACING_INITIALIZED = False


def setup_tracing():
    global TRACING_INITIALIZED
    if not TRACING_INITIALIZED:
        TRACING_INITIALIZED = True

        resource = Resource.create({"service.name": "torchsplit-runtime"})
        trace_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(trace_provider)

        metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics")
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
        metrics.set_meter_provider(meter_provider)
