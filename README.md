# Improving Feasibility via Fast Autoencoder-Based Projections

This repository is by 
[Maria Chzhen](https://www.linkedin.com/in/mariachzhen/) and 
[Priya L. Donti](https://www.priyadonti.com)
 and contains source code to reproduce the experiments in our paper 
 ["Improving Feasibility via Fast Autoencoder-Based Projections"](https://openreview.net/pdf?id=dVlkUtsyg7).

## Abstract
<p style="text-align: justify;">
Enforcing complex (e.g., nonconvex) operational constraints is a critical challenge in real-world learning and control systems. However, existing methods struggle to efficiently enforce general classes of constraints. To address this, we propose a novel data-driven amortized approach that uses a trained autoencoder as an approximate projector to provide fast corrections to infeasible predictions. Specifically, we train an autoencoder using an adversarial objective to learn a structured, convex latent representation of the feasible set. This enables rapid correction of neural network outputs by projecting their associated latent representations onto a simple convex shape before decoding into the original feasible set. We test our approach on a diverse suite of constrained optimization and reinforcement learning problems with challenging nonconvex constraints. Results show that our method effectively enforces constraints at a low computational cost, offering a practical alternative to expensive feasibility correction techniques based on traditional solvers.
</p>

<p align="center">
  <img src="figures/fab_projection.png" alt="FAB Diagram" width="800"/>
</p>

If you find this repository helpful in your publications, please consider citing our paper.
```bash
@article{chzhen2026fab,
    title={Improving Feasibility via Fast Autoencoder-Based Projections}, 
    author={Maria Chzhen and Priya L. Donti},
    year={2026},
    journal={The Fourteenth International Conference on Learning Representations},
}
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Constrained Optimization Problems

#### Training (`training.py`)
```bash
python training.py [--shape SHAPE | --shapes_2d | --shapes_multidim]
                   [--exp_type {dim,cov,capacity,num_dec} [...]]
                   [--config CONFIG [...]]
                   [--lambda_recon FLOAT [...]] [--lambda_feas FLOAT [...]]
                   [--lambda_latent FLOAT [...]] [--lambda_geom FLOAT [...]]
                   [--lambda_hinge FLOAT [...]]
```

**Shape selection** (mutually exclusive; defaults to all 2D shapes):

| Flag | Description |
|---|---|
| `--shape SHAPE` | Single shape: `blob_with_bite`, `star_shaped`, `two_moons`, `concentric_circles`, `hyperspherical_shell_{3,5,10}d` |
| `--shapes_2d` | All 2D shapes |
| `--shapes_multidim` | All multidimensional shapes (3D/5D/10D) |

> [!TIP]
> Define your own constraint sets in `data_generation.py`.

**Experiment types** (`--exp_type`): `dim`, `cov`, `capacity`, `num_dec` (default: all)

**Lambda defaults** (each accepts multiple values; all combinations are swept):

| Flag | Default |
|---|---|
| `--lambda_recon` | `[1.5, 2.0]` |
| `--lambda_feas` | `[1.0, 1.5, 2.0]` |
| `--lambda_latent` | `[1.0, 1.5]` |
| `--lambda_geom` | `[0.025]` |
| `--lambda_hinge` | `[0.5, 1.0]` |

Output files:
```
phase1_{shape}_{exp_type}_{config}.pt
phase2_{shape}_{exp_type}_{config}_{lambda_recon}_{lambda_feas}_{lambda_latent}_{lambda_geom}_{lambda_hinge}.pt
```

Examples:
```bash
# All 2D shapes, all experiment types (default)
python training.py

# Single shape and experiment type
python training.py --shape two_moons --exp_type capacity

# Multidimensional shapes, specific configs
python training.py --shapes_multidim --exp_type dim --config 3D 5D

# Quick test with overridden lambda grid
python training.py --shape blob_with_bite --exp_type cov --lambda_recon 1.5 --lambda_feas 1.0 --lambda_latent 1.0 --lambda_hinge 0.5
```

#### Testing (`testing.py`)
```bash
python testing.py [--shape SHAPE | --shapes_2d | --shapes_multidim]
                  [--exp_type {dim,cov,capacity,num_dec} [...]]
                  [--config CONFIG [...]]
                  [--models_dir DIR] [--results_dir DIR] [--output_csv FILE]
                  [--skip_latent_eval | --skip_experiments]
                  [--penalty_nn_only]
                  [--plot_sampling [--plot_models MODEL [...]] [--plot_dir DIR] [--plot_show]]
```

Runs two sequential phases by default:

- **Phase 1 – Latent evaluation:** Samples from the latent ball for each trained model, evaluates feasibility, selects the best model per shape/exp_type/config, and writes results to a CSV (e.g. `optimal_ablation_params_all.csv`).
- **Phase 2 – Experiments:** Benchmarks FAB end-to-end, FAB post-hoc projection, and a penalty-NN baseline across QP, LP, and distance-minimization objectives. Results saved to `results/` as `.txt` and `.pkl`.

Shape/experiment/config flags are identical to `training.py`.

| Option | Default | Description |
|---|---|---|
| `--models_dir DIR` | `ablations_trained_models` | Directory with trained `.pt` files |
| `--results_dir DIR` | `results` | Output directory for experiment results |
| `--output_csv FILE` | auto-named from run tag | Path for best-model CSV |
| `--n_latent_samples N` | `500` | Latent samples per model for feasibility eval |
| `--latent_radius FLOAT` | `0.5` | Radius of the latent ball |
| `--seed INT` | `42` | Global random seed |
| `--skip_latent_eval` | — | Skip Phase 1; use an existing CSV for experiments |
| `--skip_experiments` | — | Run Phase 1 only |
| `--penalty_nn_only` | — | Run penalty-NN baseline only; no AE models required |
| `--plot_sampling` | — | Save latent-vs-decoded feasibility plots (2D only) |
| `--plot_dir DIR` | `sampling_plots` | Output directory for sampling plots |
| `--plot_show` | — | Display plots interactively in addition to saving |

Examples:
```bash
# Full pipeline on all 2D shapes
python testing.py

# Skip latent eval, use existing CSV
python testing.py --skip_latent_eval --output_csv optimal_ablation_params_all.csv

# Latent eval only with sampling plots
python testing.py --shape two_moons --exp_type capacity --skip_experiments --plot_sampling
```

#### Baselines (`baselines.py`)

`baselines.py` benchmarks classical constrained optimization methods against FAB. It runs each method across QP, LP, and distance-minimization objectives, reporting feasibility rate, solve time, and optimality gap.

**Supported methods:** `projected_gradient`, `penalty_method`, `augmented_lagrangian`, `interior_point`

**Supported objectives:** `qp`, `lp`, `distance`

To configure, set `VALID_CONSTRAINTS` and `VALID_METHODS` at the top of the file, then run:
```bash
python baselines.py
```

`VALID_CONSTRAINTS` should contain shape names whose feasibility check is implemented in `data_generation.py`. `VALID_METHODS` can be any subset of the four supported methods.



### Safe Reinforcement Learning Problems
Implementation borrowed from: [Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark](https://github.com/PKU-Alignment/Safe-Policy-Optimization) (NeurIPS, 2023).

More information coming soon!
