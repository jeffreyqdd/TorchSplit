# Torch Split

A pre-compiler pass that splits models based on latency measurements and a runtime to execute the split models across multiple machines.

## Installation
Please use conda since this was tested only on conda.

1. `conda create --prefix $PSCRATCH/ts310 python=3.10`
2. `conda activate $PSCRATCH/ts310`
3. `conda install poetry`
4. `conda install graphviz`
4. `poetry install --no-root --all-groups`

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
[22:19:47] INFO     Pipeline Throughput: 682.36
           INFO     Leximin Model Throughputs:
           INFO      B: 682.36
           INFO      A: 783.12
           INFO      C: 231513.64
           INFO     Assignments:
           INFO      - Model B assigned 1x to GPU0 with memory slice 10GB
           INFO      - Model B assigned 1x to GPU1 with memory slice 10GB
           INFO      - Model B assigned 2x to GPU3 with memory slice 10GB
           INFO      - Model A assigned 5x to GPU2 with memory slice 4GB
           INFO      - Model C assigned 6x to GPU0 with memory slice 5GB
           INFO      - Model C assigned 6x to GPU1 with memory slice 5GB
           INFO      - Model C assigned 4x to GPU2 with memory slice 5GB
           INFO      - Model C assigned 4x to GPU3 with memory slice 5GB
           INFO     Node Utilization:
           INFO     Nodes: [0, 1, 2, 3]
           INFO      - GPU0: 110.0%
           INFO      - GPU1: 110.0%
           INFO      - GPU2: 205.0%
           INFO      - GPU3: 140.0%
```

This is because Model B has ~50% GPU utilization, Model A has ~30% GPU utilization, and Model C has ~10% GPU utilization.

### Serving
Prerequisite: Export batch size 1

To serve the model using Ray Serve, run the following command:

```bash
serve run examples.clip.optimized_server:app
```


## Docker and Tracing
Optionally, this repo provides a tracing hook to export traces to an OpenTelemetry collector. You can install signoz [here](https://github.com/SigNoz/signoz.git). 

```bash
cd signoz/deploy/docker
docker compose up -d
```