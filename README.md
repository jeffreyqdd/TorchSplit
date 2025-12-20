# Torch Split

A pre-compiler pass that splits models based on latency measurements and a runtime to execute the split models across multiple machines.

## Docker and Tracing
Optionally, this repo provides a tracing hook to export traces to an OpenTelemetry collector. You can install signoz [here](https://github.com/SigNoz/signoz.git). 

```bash
cd signoz/deploy/docker
docker compose up -d
```