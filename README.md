# RSCE

Research code release for **Residual Stream Context Encoding (RSCE)**, a training-free method for reusing long-context information by encoding a document into a residual-stream vector and reinjecting that representation at inference time.

RSCE replaces repeated full-context prefills with a compact latent representation and, when needed, a small explicit fact block. The code in this repository supports the experiments reported in our paper on:

- multi-document question answering on **LongBench**
- cross-file code completion on **RepoBench-C**
- architecture-specific layer calibration for residual-stream injection

## What This Release Contains

This repository is the **RSCE code release**. It includes the code needed to run RSCE itself and reproduce the main RSCE experiments.

Included:

- `RSCE/`  
  Core RSCE implementation, local benchmarking code, layer calibration, fact-block construction, and evaluation utilities.

- `RSCE_modal/`  
  Modal-native Torch/Transformers version of the RSCE pipeline for large-GPU runs and artifact-scale evaluation.

- `scripts/`  
  Helper scripts for summarizing benchmark outputs.

- `rsce_paper.tex`  
  Paper source used alongside this codebase.

Not included in the attached research-code artifact:

- external comparison baselines and their reimplementations
- third-party benchmark ports that are not part of RSCE itself
- downloaded model weights, dataset caches, or full result dumps

In other words, this release is centered on **our method**, not on distributing the full code for every comparison we ran.

## Repository Structure

At a high level, the code is organized around two execution environments.

### `RSCE/`

The core package contains:

- residual-stream extraction and injection
- model-family-specific layer selection heuristics
- LongBench QA benchmarking
- RepoBench-C benchmarking
- RSCE ablations such as vector-only vs. vector-plus-fact-block conditions

Important entrypoints:

- `calibrate_layers.py`  
  Accuracy-based sweep for choosing the injection layer `f(M)`.

- `benchmark_runner.py`  
  LongBench QA evaluation.

- `benchmark_repobench.py`  
  RepoBench-C evaluation.

### `RSCE_modal/`

The Modal-native package mirrors the RSCE pipeline for GPU-backed remote execution and larger benchmark sweeps. It uses Torch/Transformers instead of the local MLX backend and is the recommended path when reproducing larger runs on server GPUs.

## Setup

### Local RSCE package

From the repository root:

```bash
cd RSCE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This path is intended for local experimentation, layer calibration, and smaller benchmark runs.

### Modal RSCE package

For Modal-backed runs:

```bash
cd RSCE_modal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will also need:

- a configured Modal account
- Hugging Face access for the required model checkpoints and datasets
- sufficient GPU capacity for the selected model family

## Quick Start

### 1. Calibrate injection layers

RSCE relies on an empirically chosen injection layer `f(M)` for each model family.

```bash
cd RSCE
source .venv/bin/activate
python calibrate_layers.py --models llama3-8b qwen25-7b mistral-24b deepseek-14b
```

This writes calibration outputs to `calibration_results.json`.

### 2. Run LongBench QA

```bash
cd RSCE
source .venv/bin/activate
python benchmark_runner.py --models llama3-8b --tasks hotpotqa 2wikimqa --n 108
```

This evaluates baseline and RSCE conditions on the QA benchmark.

### 3. Run RepoBench-C

```bash
cd RSCE
source .venv/bin/activate
python benchmark_repobench.py \
  --models llama3-8b qwen25-7b mistral-24b deepseek-14b \
  --n 200 \
  --split cross_file_first \
  --level 2k
```

This evaluates RSCE on repository-level next-line code completion.

## RSCE Conditions

The exact set of runnable conditions depends on the benchmark entrypoint, but the core paper-facing RSCE conditions are:

- `baseline`  
  Full explicit context in the token stream.

- `vector`  
  Context compressed into a residual-stream vector only.

- `vector_f`  
  Residual-stream vector plus a compact explicit fact block.

- `vector_f_sum`  
  Residual-stream vector plus a generated summary-style fact block.

These conditions correspond to the central question in the paper: how much of the original performance can be retained after replacing most of the tokenized context with a persistent latent representation.

## Datasets

The RSCE experiments use:

- **LongBench** for multi-document QA
  - `hotpotqa`
  - `2wikimqa`

- **RepoBench-C** for cross-file code completion
  - `cross_file_first` split in the main paper runs

Dataset loading is handled inside the benchmark entrypoints. Some scripts support fallback or cached data for smoke testing, but paper-facing runs should use the actual benchmark data.

## Reproducibility Notes

- The codebase contains both local and Modal-native paths because we used different environments for different scales of experimentation.
- Layer calibration is model-family-dependent and should be run or verified before reproducing a new model result.
- RepoBench-C results depend on the chosen split and context-length bucket.
- Some large-model runs require remote GPU execution rather than local inference.

## What This README Does Not Cover

This README is intentionally scoped to the **RSCE artifact**. It does not document:

- external baseline implementations
- every exploratory script used during development
- downloaded benchmark outputs or internal result archives

Those materials were useful during the research process, but they are not part of the core RSCE release.
