# Torch Split

A pre-compiler pass that splits models based on latency measurements and a runtime to execute the split models across multiple machines.

## Installation
Please use conda since this was tested only on conda.

1. `conda create --prefix $PSCRATCH/ts310 python=3.10`
2. `conda activate $PSCRATCH/ts310`
3. `conda install poetry`
4. `conda install graphviz`
5. `poetry install --no-root --all-groups`
6. `export PYTHONPATH=$(pwd):$PYTHONPATH`

## Running
### Exporting
To export a model, run the following command:
```bash
python3 ./torch_split/cli.py export examples.clip.interface:ClipInterface -b 1
```
This will trace a batch size of 1 and export the model to `/dev/shm/switchboard.tspartd`. You can look at `/dev/shm/switchboard.tspartd/structure.json` to see the dataflow graph. To test the exported model, run:

```bash
python3 ./examples/clip/main.py
```

This will do a topological sort of the exported switchboard and run the components in order. We see that the values are identical to the original model. 

### Optimizing
To optimize a model, run the following command. Currently WIP. 
1. optimize will stop when **(A)** soft memory limit for DRAM is reached or **(B)** max batch size is reached.
2. Currently, only batch size of 1 will be optimized. Future updates will combine various batch sizes to find a better solution.

```bash
python3 ./torch_split/cli.py optimize examples.clip.interface:ClipInterface --max-batch 8
```

You can pass the "-o" flag to specify the output directory for the optimization artifacts. By default, it will be saved to `./ts_bin`. 

You can pass the "-s" flag to generate the intermediate dataflow graphs used during partitioning. It will be saved under the "visualizations" folder in the output directory.

The allocation result you get might be differ from run to run due to the non-deterministic nature of pynvml's GPU utilization. For one run: 

```
[00:09:49] INFO     Pipeline Throughput: 3050.05
           INFO     Leximin Model Throughputs:
           INFO      A: 3050.05
           INFO      B: 3172.48
           INFO      C: 11520.80
           INFO     Assignments:
           INFO      - Model B assigned 10x to GPU1 with memory slice 4GB
           INFO      - Model B assigned 9x to GPU3 with memory slice 4GB
           INFO      - Model A assigned 10x to GPU0 with memory slice 4GB
           INFO      - Model A assigned 10x to GPU2 with memory slice 4GB
           INFO      - Model C assigned 1x to GPU3 with memory slice 4GB
```

This is because Model B has ~50% GPU utilization, Model A has ~30% GPU utilization, and Model C has ~10% GPU utilization.

### Serving
Prerequisite: Export batch size 1

To serve the model using Ray Serve, run the following command:

```bash
serve run examples.clip.optimized_server:app

python3 ./examples/clip/client.py
```

or `serve run examples.clip.monolithic_server:app` to run the monolithic server.


Profiled data in `examples/clip_data` can be used to visualize latency and GPU utilization. To generate the plots, run:

```
Performance Improvement Summary:
QPS        Latency Improvement (%)   Throughput Improvement (%)
------------------------------------------------------------
1          48.18                     1.68
2          49.26                     1.68
4          50.94                     1.66
8          50.62                     1.79
16         52.07                     0.52
32         52.23                     1.13
64         54.91                     1.38
128        67.15                     3.46
256        90.02                     43.13
512        63.77                     76.19
1024       38.23                     58.60
2048       37.67                     66.21
------------------------------------------------------------
```


## Docker and Tracing
Optionally, this repo provides a tracing hook to export traces to an OpenTelemetry collector. You can install signoz [here](https://github.com/SigNoz/signoz.git). 

```bash
cd signoz/deploy/docker
docker compose up -d
```