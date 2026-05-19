# Learning-Augmented Greedy Sparse Recovery via Operator-Aware Support Union

This repository contains code, experiments, figures, and a NeurIPS-style paper draft for a learning-augmented sparse recovery project.

The main method is **OMP-LearnedUnion**, a hybrid sparse recovery algorithm that uses a learned operator-aware support predictor as a correction layer for Orthogonal Matching Pursuit (OMP). Instead of replacing classical greedy recovery with an end-to-end neural network, the method keeps the OMP backbone, adds learned support candidates, performs least-squares debiasing on the union support, prunes back to the target sparsity level, and refits least squares on the final support.

## Research Question

Classical greedy sparse recovery algorithms such as OMP are fast and interpretable, but they can make early support mistakes in high-sparsity, coherent, or compressible regimes. This project asks:

> Can a learned support model correct greedy support errors without replacing the classical sparse recovery algorithm?

The core hypothesis is that learning is most useful as a **support-correction layer**: the learned model only needs to propose useful coordinates missed by OMP, while least-squares pruning filters the enlarged candidate set.

## Method: OMP-LearnedUnion

Given measurements

```math
y = Ax + \eta,
```

where `A` is an `m x n` sensing matrix with `m < n`, and `x` is sparse or compressible, the method proceeds as follows:

1. Run OMP to obtain an initial support estimate `S_OMP`.
2. Build coordinate-wise operator-aware features from marginal correlations, column norms, coherence statistics, marginal-correlation ranks, early OMP selections, and residual correlations.
3. Use a learned MLP support predictor to select learned candidates `S_L`.
4. Form the union candidate set `C = S_OMP union S_L`.
5. Perform least-squares debiasing on `C`.
6. Prune to the top-`k` coefficients.
7. Refit least squares on the final support.

This produces the final reconstruction.

## Repository Structure

```text
operator-aware-sparse-recovery/
├── src/
│   ├── algorithms/
│   │   ├── omp.py
│   │   ├── cosamp.py
│   │   ├── subspace_pursuit.py
│   │   └── iht.py
│   ├── data/
│   │   ├── make_sparse_signal.py
│   │   └── make_sensing_matrix.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── learned/
│   │   ├── feature_builder.py
│   │   └── support_mlp.py
│   └── utils/
│       └── seed.py
├── experiments/
│   ├── run_omp_union_support.py
│   ├── run_phase_transition.py
│   ├── plot_phase_transition.py
│   ├── summarize_phase_transition.py
│   ├── run_compressible_phase_transition.py
│   ├── plot_compressible_phase_transition.py
│   ├── summarize_compressible_phase_transition.py
│   ├── run_coherence_sweep.py
│   ├── plot_coherence_sweep.py
│   ├── analyze_failure_cases.py
│   ├── run_union_ablation.py
│   ├── plot_union_ablation.py
│   └── test_union_ablation_significance.py
├── results/
├── figures/
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the core dependencies manually:

```bash
pip install numpy pandas scipy matplotlib torch
```

## Main Experiments

### 1. Single OMP-LearnedUnion experiment

```bash
python experiments/run_omp_union_support.py
```

Expected output:

```text
results/learned_support_results.csv
```

### 2. Exact sparse phase-transition experiment

```bash
python experiments/run_phase_transition.py
python experiments/plot_phase_transition.py
python experiments/summarize_phase_transition.py
```

Expected outputs:

```text
results/phase_transition_results.csv
results/phase_transition_gain_summary.csv
figures/phase_transition_f1_gain_gaussian.png
figures/phase_transition_f1_gain_coherent.png
figures/phase_transition_nrmse_gain_gaussian.png
figures/phase_transition_nrmse_gain_coherent.png
```

### 3. Compressible phase-transition experiment

```bash
python experiments/run_compressible_phase_transition.py
python experiments/plot_compressible_phase_transition.py
python experiments/summarize_compressible_phase_transition.py
```

Expected outputs:

```text
results/compressible_phase_transition_results.csv
results/compressible_phase_transition_gain_summary.csv
figures/compressible_phase_transition_f1_gain_gaussian.png
figures/compressible_phase_transition_f1_gain_coherent.png
figures/compressible_phase_transition_nrmse_gain_gaussian.png
figures/compressible_phase_transition_nrmse_gain_coherent.png
```

### 4. Coherence sweep

```bash
python experiments/run_coherence_sweep.py
python experiments/plot_coherence_sweep.py
```

Expected outputs:

```text
results/coherence_sweep_results.csv
figures/coherence_sweep_f1_gain.png
figures/coherence_sweep_nrmse_gain.png
figures/coherence_sweep_all_methods_f1.png
figures/coherence_sweep_all_methods_nrmse.png
```

### 5. Failure-case analysis

```bash
python experiments/analyze_failure_cases.py
```

Expected outputs:

```text
results/failure_case_summary.csv
results/failure_case_examples.txt
```

### 6. Union ablation

```bash
python experiments/run_union_ablation.py
python experiments/plot_union_ablation.py
python experiments/test_union_ablation_significance.py
```

Expected outputs:

```text
results/union_ablation_results.csv
figures/union_ablation_f1.png
figures/union_ablation_nrmse.png
```

## Key Results

### Exact sparse phase transition

Across 18 exact sparse phase-transition settings over Gaussian and coherent sensing matrices:

- OMP-LearnedUnion improves NRMSE in **14/18** settings.
- OMP-LearnedUnion improves support F1 in **15/18** settings.

| Method | NRMSE | Support F1 |
|---|---:|---:|
| OMP-LearnedUnion | **0.1362** | **0.8668** |
| OMP | 0.1404 | 0.8622 |
| Subspace Pursuit | 0.1829 | 0.8375 |
| CoSaMP | 0.2211 | 0.7635 |
| IHT | 0.4286 | 0.5865 |
| LearnedSupportMLP | 0.5428 | 0.4720 |
| CorrelationTopK | 0.6409 | 0.4132 |

### Compressible signal phase transition

Across 18 compressible phase-transition settings:

- OMP-LearnedUnion improves NRMSE in **18/18** settings.
- OMP-LearnedUnion improves support F1 in **18/18** settings.

| Method | NRMSE | Support F1 |
|---|---:|---:|
| OMP-LearnedUnion | **0.2995** | **0.7236** |
| OMP | 0.3040 | 0.7182 |
| Subspace Pursuit | 0.3476 | 0.6966 |
| CoSaMP | 0.3405 | 0.6745 |
| IHT | 0.4799 | 0.5567 |
| LearnedSupportMLP | 0.5600 | 0.4685 |
| CorrelationTopK | 0.6610 | 0.4077 |

### Coherence sweep

The gain over OMP increases as coherence strength increases:

| Coherence Strength | NRMSE Gain | F1 Gain |
|---:|---:|---:|
| 0.25 | 0.004736 | 0.006018 |
| 0.35 | 0.009257 | 0.009623 |
| 0.50 | 0.029490 | 0.025133 |

This supports the hypothesis that learned support correction is most useful when coherent matrix geometry makes greedy support selection ambiguous.

### Failure-case analysis

In a high-coherence compressible setting with `n=256`, `m=128`, `k=58`, and coherence strength `0.50`:

| Statistic | Value |
|---|---:|
| Average NRMSE gain | 0.028585 |
| Average F1 gain | 0.024741 |
| Trials with NRMSE improvement | 152 / 200 |
| Trials with F1 improvement | 146 / 200 |
| Average true support coordinates missed by OMP | 27.505 |
| Average missed OMP coordinates recovered by learned support | 6.150 |
| Average missed OMP coordinates retained after union pruning | 2.495 |
| Average false OMP coordinates removed by pruning | 7.005 |

### Union ablation

In the high-coherence compressible setting:

| Method | NRMSE | Support F1 |
|---|---:|---:|
| OMP | 0.6104 | 0.5184 |
| OMP + Random Union | 0.6512 | 0.5085 |
| OMP + Correlation Union | 0.6214 | 0.5331 |
| OMP-LearnedUnion | **0.5748** | **0.5433** |

The ablation shows that the improvement is not merely due to enlarging the candidate set. Random union worsens performance, correlation union improves F1 but worsens NRMSE, while learned union improves both metrics.

## Main Figures

Use the following figures in the paper:

```text
phase_transition_f1_gain_gaussian.png
phase_transition_f1_gain_coherent.png
phase_transition_nrmse_gain_gaussian.png
phase_transition_nrmse_gain_coherent.png

compressible_phase_transition_f1_gain_gaussian.png
compressible_phase_transition_f1_gain_coherent.png
compressible_phase_transition_nrmse_gain_gaussian.png
compressible_phase_transition_nrmse_gain_coherent.png

coherence_sweep_f1_gain.png
coherence_sweep_nrmse_gain.png
coherence_sweep_all_methods_f1.png
coherence_sweep_all_methods_nrmse.png

union_ablation_f1.png
union_ablation_nrmse.png
```

## Interpretation

The empirical results suggest that learned support prediction is not strong enough to replace classical sparse recovery by itself. However, it is useful as a correction layer on top of OMP.

The key mechanism is:

1. OMP makes support mistakes in hard regimes.
2. The learned predictor proposes some true coordinates that OMP misses.
3. Unioning the OMP and learned supports gives least squares a better candidate set.
4. Least-squares pruning removes many false OMP coordinates.
5. The final support improves reconstruction error and support F1.

## Current Limitations

This project is currently a preliminary research draft. Important limitations include:

- The learned model is a simple coordinate-wise MLP.
- Experiments focus on synthetic Gaussian and coherent sensing matrices.
- The sparsity level `k` is assumed known.
- The current theory is informal.
- Stronger learned baselines such as LISTA, LAMP, unfolded ISTA, and AMP-inspired neural networks are not yet included.
- More structured operators such as partial Fourier, Hadamard, block-correlated dictionaries, and real inverse-problem operators should be tested.

## Future Work

Useful next steps include:

1. Add a formal theorem for the support-correction condition.
2. Add learned baselines such as LISTA and LAMP.
3. Add structured sensing matrices:
   - partial Fourier,
   - Hadamard,
   - ill-conditioned Gaussian,
   - block-correlated dictionaries.
4. Expand the phase-transition grid:
   - `n in {256, 512, 1024}`,
   - `m/n in {0.25, 0.375, 0.5, 0.625}`,
   - `k/m in {0.2, 0.3, 0.4, 0.5}`.
5. Study robustness to unknown sparsity `k`.
6. Test on real inverse-problem operators.

## Suggested Citation

```bibtex
@misc{choi2026omplearnedunion,
  title        = {Learning-Augmented Greedy Sparse Recovery via Operator-Aware Support Union},
  author       = {Choi, Taewoon},
  year         = {2026},
  note         = {Preliminary research manuscript}
}
```

## Status

This project is currently a **preliminary research manuscript** intended for professor feedback and further development. The current version is likely most suitable for:

- professor research feedback,
- an internal research report,
- a NeurIPS/ICML/ICLR workshop-style draft after polishing,
- future AISTATS or main-conference submission after additional theory, baselines, and experiments.
