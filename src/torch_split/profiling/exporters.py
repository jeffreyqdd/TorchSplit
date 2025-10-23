import io
import json
import sys

import msgpack  # type: ignore
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class FileTraceBytesExporter(SpanExporter):
    """Binary file exporter for OpenTelemetry spans using Msgpack format"""

    def __init__(self, writable_fh: io.BytesIO):
        self._fh = writable_fh
        self._packer = msgpack.Packer(use_bin_type=True)

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                # span_json = json.loads(span.to_json())
                packed = self._packer.pack(span.to_json())
                self._fh.write(packed)
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"Error writing spans: {e}")
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._fh.flush()
        return super().force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._fh.flush()
        self._fh.close()


class ConsoleExporter(SpanExporter):
    def __init__(self):
        self._fh = sys.stdout

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                span_json = json.loads(span.to_json())
                self._fh.write(json.dumps(span_json, indent=2))
            return SpanExportResult.SUCCESS
        except Exception as e:
            print(f"Error writing spans: {e}")
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._fh.flush()
        return super().force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._fh.flush()
        self._fh.close()
