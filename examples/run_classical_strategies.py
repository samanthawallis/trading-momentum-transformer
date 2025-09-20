import os
from mom_trans.backtest import run_classical_methods

INTERVALS = [(2000, y, y + 1) for y in range(2024, 2025)]

REFERENCE_EXPERIMENT = "experiment_quandl_100assets_lstm_cpnone_len63_notime_div_v1"

features_file_path = os.path.join(
    "data",
    "quandl_cpd_nonelbw.parquet",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
