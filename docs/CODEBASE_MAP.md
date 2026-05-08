# Codebase Map

This document maps major files and explains how the repository is wired.

## 1) High-Level Pipeline

1. Build environment + force models (`HelperFunctions/PropFuncs.py`).
2. Build observation sets (simulated or real; `HelperFunctions/ObsFunc.py`, `HelperFunctions/nsdc.py`).
3. Run estimation (`ObservationImplementation.py`, `Estimation_Main.py`).
4. Save arrays and metadata to results folders.
5. Post-process (`MainPostprocessing.py`) and compare variants.
6. Explore interactively (`DashInteractivePlotFull.py`) and export publication figures (`MatplotlibExport.py`).

## 2) Root Scripts by Responsibility

## Estimation/Propagation Runners

- `RunSinglePropagation.py`
  - Propagates Triton only.
  - Includes timestep-sensitivity analysis sections.
  - Writes state/dependent variable arrays and PDF plots.

- `Estimation_Main.py`
  - Simulated-observation estimation baseline.
  - Generates pseudo-observations and runs estimation output creation.

- `ObservationImplementation.py`
  - Main estimation engine shared by higher-level scripts.
  - Handles both simulated and real observations.
  - Applies weighting/filtering flow and can inject manual DEC bias.

- `RunMultipleEstimations.py`
  - Defines large variant dictionaries and loops over scenarios.
  - Builds aggregate simulation dictionaries and comparison plots.

- `ConcurentEstimationRealObservations.py`
  - Real-observation experiment driver.
  - Uses template functions from `EstimationAnalysisTemplates.py`.
  - Adds weight tables, saves aggregate pickles, and creates LaTeX outputs.

## Analysis and Visualization

- `MainPostprocessing.py`
  - Loads `.npy` outputs.
  - Builds observation-level DataFrames.
  - Computes residual/timeframe summaries and plotting products.

- `Analysis_All_Estimations_and_Data.py`
  - Cross-estimation analysis utility with postprocessing imports.
  - Used for broader comparative studies.

- `DashInteractivePlot.py`
  - Earlier/simple Dash interface.

- `DashInteractivePlotFull.py`
  - Full dashboard with tabs for RSW, weights, residuals, stats, correlations, parameter updates.
  - Exposes processing helpers imported by `MatplotlibExport.py`.

- `MatplotlibExport.py`
  - Batch PDF export layer over dashboard-style data structures.
  - Driven by presets from `ExportConfigs/*.py`.
  - Also writes LaTeX tables for selected analyses.

## Domain/Support Scripts

- `NSDC_processing.py`
  - NSDC-oriented processing workflow (input parsing, plotting, helper usage).

- `PoleInvestigation.py`
  - Pole-focused propagation/analysis script.

- `ParserTestFile.py`
  - Observation parsing testbed and parser extract generation.

- `Test_Observations.py`, `ObservationsTestFileIgnore.py`
  - Observation analysis/testing scripts and plotting experiments.

## 3) HelperFunctions Package

- `HelperFunctions/PropFuncs.py`
  - `Create_Env`: body settings + system creation.
  - `Create_Acceleration_Models`: acceleration model assembly.
  - `Create_Propagator_Settings`: propagator setup.
  - `make_relative_position_pseudo_observations`: pseudo-observation generation.
  - `Create_Estimation_Output`: estimation solve orchestration.
  - `PoleModel`: parameterized Neptune pole model generation.

- `HelperFunctions/ObsFunc.py`
  - Real-observation loading and station handling.
  - SPICE residual computation for observation sets.
  - Weight assignment methods and timeframe splitting.
  - Observation bias application tools.

- `HelperFunctions/ProcessingUtils.py`
  - Residual history formatting.
  - Inertial↔RSW rotations for vectors/covariances.
  - State history conversion helpers.

- `HelperFunctions/FigUtils.py`
  - Common plotting utilities used across all runners.
  - RSW/FFT/PSD/residual/correlation/parameter update plotting.

- `HelperFunctions/nsdc.py`
  - NSDC text parsing and conversion routines.
  - Observatory metadata and frame/time normalization.
  - Observation collection creation and outlier filtering.

## 4) Export Configuration Presets

- `ExportConfigs/CASE1_Manual_Bias.py`
  - Pole-estimation figure pack (IAU vs fitted-pole variants).

- `ExportConfigs/WeightAnalysis.py`
  - Weight-scheme comparison figure pack.

- `ExportConfigs/WeightAnalysis_Pole.py`
  - Weight-scheme comparison for pole-estimation cases.

- `ExportConfigs/WeightComparison.py`
  - Cross-dataset weight analysis comparison.

- `ExportConfigs/ObservationalDataset.py`
  - Observation catalog summary figures and LaTeX table settings.

- `ExportConfigs/SimObs_ParameterAnalysis.py`
  - Simulated-observation parameter analysis exports.

## 5) Data and Metadata Files

- `file_names.json`
  - Observation file IDs included by default in many scripts.

- `observation_set_ids.json`
  - Observation set index metadata.

- `summary.txt`
  - Tabular weighting/summary input consumed in some runs.

- `Observations/AllModernECLIPJ2000/*.csv`
  - Processed observation datasets used by real-observation workflows.

- `Observations/NeptuneObservations/*.txt`
  - Raw NSDC observations (absolute-type sets).

- `Observations/RawRelativeObservations/*.txt`
  - Raw NSDC relative-observation sets.

- `Observations/Observatories.txt`
  - Observatory code metadata.

## 6) Important Execution Behavior

Several top-level scripts are not pure libraries and execute work immediately when run/imported (variant loops, plotting blocks, pickle writing).

Before launching long runs, always check:

1. Hardcoded result/output directories.
2. `runSim`/toggle booleans.
3. Variant dictionaries and selected parameter sets.
4. Required kernel/data file presence.

## 7) Typical End-to-End Run Order

1. `RunSinglePropagation.py` for baseline sanity checks.
2. `Estimation_Main.py` (simulated observations) or `ConcurentEstimationRealObservations.py` (real observations).
3. `MainPostprocessing.py` to inspect residual and weight behavior.
4. `DashInteractivePlotFull.py` for interactive comparative analysis.
5. `MatplotlibExport.py` with selected `ExportConfigs/*.py` for final figure export.