"""Export configuration: WeightAnalysis_Old
Same figures as WeightAnalysis but pointing to the old dataset,
with '(Old)' appended to all figure titles.
"""

from ExportConfigs import WeightAnalysis as _base

DATASET_LABEL   = 'WeightAnalysis_Old'  # TODO: set matching key in DashInteractivePlotFull DATA_FILES
SELECTED_SIMS   = _base.SELECTED_SIMS
SIM_LABELS      = _base.SIM_LABELS
OUTPUT_DIR      = _base.OUTPUT_DIR
SINGLE_SIM_TABLES = _base.SINGLE_SIM_TABLES
SIM_COLORS      = _base.SIM_COLORS
SIM_LINESTYLE   = _base.SIM_LINESTYLE
SIM_MARKERS     = _base.SIM_MARKERS

FIGURES_TO_EXPORT = []
for _fn_name, _kwargs in _base.FIGURES_TO_EXPORT:
    _new_kwargs = dict(_kwargs)
    if 'title' in _new_kwargs:
        _new_kwargs['title'] += ' (Old)'
    FIGURES_TO_EXPORT.append((_fn_name, _new_kwargs))
