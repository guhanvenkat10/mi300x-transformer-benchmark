# MI300X transformer throughput benchmark

A single-file benchmark to measure real training throughput on an MI300X (or any CUDA/ROCm GPU), so
compute requirements can be extrapolated from a measured number instead of an estimate.

It also doubles as a stack smoke test: if it runs clean, then PyTorch + ROCm, bf16 autocast,
attention, backward pass, and AdamW all work on the node.

## Requirements

`torch` only. Nothing else.

## Run it

```bash
python smoke_test.py --preset large
```

That's the ~200M parameter model, the largest of the four sizes and the only one that costs real time.

The MI300X has ~192 GB per GPU, so the default batch size of 16 is very conservative. Push it up for
better utilization and higher throughput:

```bash
python smoke_test.py --preset large --batch-size 64
```

To get the full picture, run all four model sizes:

```bash
python smoke_test.py --preset tiny
python smoke_test.py --preset small
python smoke_test.py --preset medium
python smoke_test.py --preset large
```

`tiny` and `small` finish almost instantly, which is the point: most runs in the planned sweep are
those small presets.

## What to look at

Two lines in the output:

- `time for N steps` and `ms/step`
- `throughput ... tokens/sec`

The script then prints an extrapolation table converting tokens/sec into hours-per-epoch on 1 GPU and
on 8 GPUs, for a range of possible corpus sizes.

## What it actually does

Trains a decoder-style transformer on synthetic data (random tokens, realistic vocab and sequence
length). No real dataset is needed, so it runs anywhere immediately.

The synthetic data is not the point. The compute shape is: parameter count, sequence length, batch
size, and a real forward + backward + optimizer step. Those match the intended workload, so the
measured tokens/sec transfers.

Loss will sit near `ln(vocab)` (~4.85) because the data is random. That is expected and is only a
sanity check that the training loop is wired up correctly.

## Options

```
--preset      tiny | small | medium | large   (default: large)
--seq-len     tokens per sample               (default: 1024)
--batch-size  samples per step                (default: 16)
--steps       timed steps                     (default: 200)
--warmup      untimed warmup steps            (default: 10)
--dtype       bf16 | fp16 | fp32              (default: bf16)
--list-presets
```
