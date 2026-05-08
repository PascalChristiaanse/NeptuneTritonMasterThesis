# Neptune Triton Master Thesis Codebase

This repository contains the simulation, estimation, and analysis tooling used for orbit and pole-model studies of Triton around Neptune.

The project is built around TudatPy-based propagation and estimation, with both simulated and real astrometric observations, followed by plotting/export pipelines for thesis-quality figures.

## What This Repository Does

The codebase supports three main workflows:

1. Propagate Triton dynamics under configurable force models.
2. Estimate model parameters from simulated or real observations.
3. Post-process and visualize residuals, formal errors, parameter updates, and correlation structures.

Typical estimated parameter sets include:

- Initial state.
- Pole position/rate terms.
- Pole libration terms.
- Optional gravity-related parameters in selected experiments.

## Repository Structure

- `HelperFunctions/`
	- Shared utilities for environment setup, propagators, observation loading/weighting, frame rotations, and plotting.
- `ExportConfigs/`
	- Figure-export presets for `MatplotlibExport.py`.
- `Observations/`
	- Processed observation CSV files and raw NSDC text files.
- Root-level orchestration scripts
	- Estimation runners, analysis scripts, interactive dashboards, and publication export tools.

For a detailed file-by-file map, see `docs/CODEBASE_MAP.md`.

## Core Scripts (Most Useful Entry Points)

- `RunSinglePropagation.py`
	- Single propagation runs and timestep comparison experiments.
- `Estimation_Main.py`
	- Simulated-observation estimation pipeline.
- `ObservationImplementation.py`
	- Main estimation implementation used by multiple runners/templates.
- `ConcurentEstimationRealObservations.py`
	- Real-observation study driver with template-based experiment variants.
- `MainPostprocessing.py`
	- Loading output arrays and generating summary/statistical plots.
- `DashInteractivePlotFull.py`
	- Interactive dashboard over precomputed simulation dictionaries.
- `MatplotlibExport.py`
	- Batch export of publication-ready PDFs from dashboard datasets.

## Environment and Dependencies

This project assumes a Python environment with scientific and astro-dynamics tooling.

Minimum practical dependency set inferred from imports:

- `tudatpy`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `plotly`
- `dash`
- `pyyaml`
- `astropy`

### External Data/Kernels

Many scripts expect SPICE kernels under:

- `Kernels/`

Commonly referenced files include:

- `pck00010.tpc`
- `gm_de440.tpc`
- `nep097.bsp`
- `naif0012.tls`

If those are missing, propagation/estimation scripts will fail during kernel load.

## Quick Start

From repository root:

```bash
python RunSinglePropagation.py
```

Then move to estimation workflows:

```bash
python Estimation_Main.py
python ConcurentEstimationRealObservations.py
```

For interactive analysis:

```bash
python DashInteractivePlotFull.py
```

For PDF export with a selected config:

1. Set `ACTIVE_CONFIG` in `MatplotlibExport.py`.
2. Run:

```bash
python MatplotlibExport.py
```

## Configuration Pattern Used Across Scripts

Most runners construct a nested settings dictionary:

- `settings["env"]`
	- Bodies, frame origin/orientation, epochs, Neptune rotation model option.
- `settings["acc"]`
	- Propagation bodies, central body, acceleration model choices.
- `settings["prop"]`
	- Start/end epochs, initial epoch, step size, propagator configuration.
- `settings["obs"]`
	- Simulated vs real observations, source files, weighting/filtering options.
- `settings["est"]`
	- Estimated parameter list and estimation-specific options.

This common structure is consumed by helper functions in `HelperFunctions/PropFuncs.py`, `HelperFunctions/ObsFunc.py`, and downstream estimation modules.

## Outputs You Will See

Depending on script/workflow, output folders usually contain:

- `settings.yaml`
- `state_history*.npy`
- `residuals*.npy`
- `correlations.npy`
- `parameter_history.npy`
- `observation_weights.csv`
- Figures (`.pdf`)
- Aggregated dictionaries (`simulations.pkl`, `simulations_with_weights.pkl`)

## Known Caveats

1. Many orchestration scripts run code at module import time (not strictly behind `if __name__ == "__main__"`).
2. Several paths are hardcoded to local result directories in analysis sections.
3. Experiment activation is controlled by booleans and inline variant dictionaries; review these before running expensive jobs.
4. `summary.txt` in repository root is currently used as a tabular data input in at least one workflow, not a narrative summary.

## Recommended First Workflow for New Users

1. Run a short propagation in `RunSinglePropagation.py`.
2. Run a minimal estimation variant in `ConcurentEstimationRealObservations.py` or `Estimation_Main.py`.
3. Load generated arrays with `MainPostprocessing.py`.
4. Inspect comparisons in `DashInteractivePlotFull.py`.
5. Export thesis figures with `MatplotlibExport.py` + appropriate `ExportConfigs/*.py` preset.

## Additional Documentation

- `docs/CODEBASE_MAP.md` for module-level responsibilities and workflow routing.
