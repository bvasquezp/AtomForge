# AtomForge &mdash; Deep-Material v2

**Generative discovery of Metal-Organic Frameworks for green hydrogen storage using Riemannian Flow Matching, Grand Canonical Monte Carlo screening, and semi-empirical DFT validation.**

---

## Overview

AtomForge is an end-to-end computational pipeline that learns the distribution of crystalline MOF structures from the [CoREMOF 2019](https://zenodo.org/record/3677685) database and generates novel candidates optimized for gas adsorption. The system operates in four stages:

1. **Learn** -- A SE(3)-equivariant GNN learns to map noise to valid crystal structures via Conditional Flow Matching on the Riemannian manifold SPD(3).
2. **Generate** -- New MOF candidates are sampled by integrating the learned ODE from t=0 (noise) to t=1 (crystal), producing atomic coordinates, species, and unit cell parameters simultaneously.
3. **Screen** -- Candidates are relaxed with Lennard-Jones force fields and evaluated for CO2 uptake capacity using a built-in Grand Canonical Monte Carlo engine.
4. **Validate** -- Top candidates undergo GFN2-xTB semi-empirical relaxation and H2 adsorption energy calculation.

## Architecture

```
CoREMOF CIFs
    |
    v
[dataset.py] -- CIF -> Periodic Graph (VoronoiNN bonds, fractional coords, lattice matrix)
    |
    v
[crystal_model.py] -- CrystalFlowModel
    |   AtomEncoder (Z -> dense vector)
    |   CrystalGNN (SE(3) message passing with spherical harmonics l=0,1)
    |   Conservative field v = -grad(Phi) via autograd
    |   Virial stress tensor for lattice dynamics
    |
    v
[train_flow.py] -- Conditional Flow Matching
    |   Sinkhorn OT loss with t^4 annealing (per-graph, O(N_i^2))
    |   Hutchinson divergence estimator (Liouville theorem)
    |   Log-Euclidean lattice loss on Sym(3)
    |   Gradient clipping + divergence clamping [-10, 10]
    |
    v
[generate_flow.py] -- ODE Integration (RK4 / dopri5)
    |   LogEuclideanExp: SPD(3) diffeomorphism with Daleckii-Krein backward
    |   Adaptive density-based volume correction
    |   WyckoffGuidedRecorder for trajectory capture
    |   CIF/XYZ export with full unit cell
    |
    v
[run_gcmc_analysis.py] -- GCMC Adsorption Screening
    |   Pure-Python LJ engine (UFF + TraPPE mixing rules)
    |   Insert / Delete / Translate moves with Metropolis acceptance
    |   Structural validation (overlap detection)
    |   RASPA2 input file generation
    |
    v
[validate_xtb.py] -- DFT Validation
    |   GFN2-xTB relaxation via ASE + tblite
    |   E_ads(H2) = E(MOF+H2) - E(MOF) - E(H2)
    |   Parallel execution (ProcessPoolExecutor)
    |
    v
  Ranked candidates -> results/xtb_validation_report.csv
```

## Key Mathematical Components

| Component | Implementation | Purpose |
|---|---|---|
| **LogEuclideanExp** | Custom `torch.autograd.Function` | Maps Sym(3) -> SPD(3) via matrix exponential with Taylor-regularized backward for degenerate eigenvalues |
| **Sinkhorn-Knopp OT** | Per-graph with MIC wrapping | Wasserstein-proxy loss on the flat torus T^3 for fractional coordinate matching |
| **Hutchinson Divergence** | Stochastic trace estimator | Enforces Liouville's theorem (volume preservation) in the CNF |
| **Virial Stress Tensor** | Learned force head + dyadic product | Drives lattice parameter evolution during generation |
| **Conservative Field** | v = -grad(Phi) via `torch.autograd` | Guarantees energy-consistent coordinate dynamics |

## Project Structure

```
Deep-Material/
├── config.yaml                  # Central configuration (model, training, paths)
├── requirements.txt             # Python dependencies
├── tracker.html                 # Interactive project dashboard
├── TASKS.md                     # Development roadmap
│
├── src/
│   ├── utils.py                 # Chemical constants, I/O, LogEuclideanExp, MIC distance
│   ├── dataset.py               # CIF -> periodic graph pipeline (VoronoiNN, PBC)
│   ├── crystal_model.py         # CrystalFlowModel (SE(3) GNN + Flow Matching heads)
│   ├── train_flow.py            # Training loop (Sinkhorn OT, Hutchinson div, logging)
│   ├── generate_flow.py         # ODE integration, CNF dynamics, CIF/XYZ export
│   ├── generate_final_batch.py  # Batch generation of 20 candidates (CLI)
│   ├── relax_structure.py       # Quick LJ relaxation via ASE
│   ├── run_gcmc_analysis.py     # GCMC engine + RASPA input generator (771 lines)
│   ├── run_batch_gcmc.py        # Orchestrates relax -> GCMC for all candidates
│   ├── run_porosity_analysis.py # Zeo++ porosity characterization (ASA, AV, Di, Df)
│   ├── validate_xtb.py          # GFN2-xTB relaxation + H2 adsorption energy
│   ├── export_cinematic_xyz.py  # Trajectory T^3 -> Cartesian for Blender
│   ├── blender_animator.py      # Blender material driver for crystallization movies
│   ├── plot_crystallization.py  # Training metrics visualization
│   └── unzip_mofs.py            # CoREMOF dataset extraction
│
├── data/
│   ├── raw_cifs/                # Original CIF files from CoREMOF
│   └── processed_v2/           # Serialized periodic graphs (.pt)
│
├── models/
│   ├── flow_best.pth            # Best validation checkpoint (val loss: 0.6266)
│   ├── flow_final.pth           # Final epoch checkpoint
│   └── checkpoints/             # Periodic checkpoints (every 50 epochs)
│
└── results/
    ├── candidates/              # 20 generated MOF CIFs + LJ-relaxed variants
    ├── gcmc/
    │   └── gcmc_results.csv     # GCMC adsorption screening results
    └── xtb_validated/           # GFN2-xTB relaxed structures (pending)
```

## Training Results

| Metric | Value |
|---|---|
| Dataset | CoREMOF 2019 (9,784 train / 1,223 val) |
| Architecture | SchNet backbone, 64 hidden dim, 3 layers, ~100K params |
| Training | 500 epochs on 8 CPU cores (Xeon) |
| Best Val Loss | **0.6266** |
| Candidates Generated | 20 MOFs (RK4, 100 integration steps) |

## GCMC Screening Results (excerpt)

| Candidate | Loading (mg/g) | Loading (cm3 STP/cm3) | Henry (mol/kg/Pa) |
|---|---|---|---|
| mof_candidate_014 | **48.02** | 27.10 | 1.09e-05 |
| mof_candidate_006 | 23.44 | 13.23 | 5.33e-06 |
| mof_candidate_001 | 23.37 | 13.19 | 5.31e-06 |
| mof_candidate_002 | 11.75 | 6.63 | 2.67e-06 |
| mof_candidate_004 | 6.60 | 3.72 | 1.50e-06 |

*CO2 at 298 K, 1 bar. 5000 production cycles, 2000 equilibration.*

## Quick Start

```bash
# 1. Setup environment
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
python src/unzip_mofs.py
python src/dataset.py --action all

# 3. Train model
python -u src/train_flow.py --config config.yaml

# 4. Generate candidates
python src/generate_final_batch.py --num_samples 20

# 5. Screen for gas adsorption (relax + GCMC)
python src/run_batch_gcmc.py

# 6. DFT validation of top candidates (requires tblite)
python src/validate_xtb.py
```

## Dependencies

- **Core:** PyTorch >= 2.0, PyTorch Geometric >= 2.4, e3nn
- **Chemistry:** pymatgen, ASE, tblite (optional, for DFT)
- **Tracking:** Weights & Biases

## References

| Paper | Venue | Relevance |
|---|---|---|
| MOFFlow | ICLR 2025 | Riemannian Flow Matching on SE(3) for MOFs |
| FlowMM | ICML 2024 | Riemannian FM for crystal generation |
| CDVAE | ICLR 2022 | Diffusion model baseline for crystals |

## License

Academic project.
