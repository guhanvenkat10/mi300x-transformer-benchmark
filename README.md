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

First, the backend line near the top:

- `sdpa avail : flash=... efficient=... math=...` shows which attention kernels this build actually
  has. On ROCm the auto-dispatcher often silently skips the fast ones, so the script probes them
  directly and then **forces** the best available.
- `attention : forcing '...'` says which one is running. `flash` or `efficient` is good (fused, no
  seq x seq matrix). `math` means both fused kernels are missing from the build and attention is
  materialized, which is memory-bound and under-reports the GPU.

Then the results:

- `time for N steps` and `ms/step`
- `throughput ... tokens/sec`
- peak GPU memory, and a line confirming whether the fused kernel engaged.

The script then prints an extrapolation table converting tokens/sec into hours-per-epoch on 1 GPU and
on 8 GPUs, for a range of possible corpus sizes.

### Attention backend

```
--attn auto        # default: force the best available (flash > efficient > math)
--attn flash       # force flash; errors if not built
--attn efficient   # force memory-efficient
--attn math        # force materialized (slow; for comparison only)
```

To see the difference directly, run the same config with `--attn flash` (or `efficient`) vs
`--attn math` and compare memory and throughput.

If the probe reports `flash=NO efficient=NO`, this PyTorch build has no fused attention kernel and is
materializing the score matrix. On ROCm/MI300X the fix is a flash-attention build for AMD (aotriton,
which recent ROCm PyTorch wheels ship, or a Composable-Kernel flash-attn install). That single change
is usually a large throughput and memory win, so it is worth sorting before the real runs.

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
--preset      tiny | small | medium | large       (default: large)
--seq-len     tokens per sample                   (default: 1024)
--batch-size  samples per step                    (default: 16)
--steps       timed steps                         (default: 200)
--warmup      untimed warmup steps                (default: 10)
--dtype       bf16 | fp16 | fp32                  (default: bf16)
--attn        auto | flash | efficient | math     (default: auto)
--list-presets
```
