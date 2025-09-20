
import os

CPD_LBWS = [10, 21, 63, 126, 256]
CPD_DEFAULT_LBW = 21
BACKTEST_AVERAGE_BASIS_POINTS = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
USE_KM_HYP_TO_INITIALISE_KC = True

CPD_QUANDL_OUTPUT_FOLDER = lambda lbw: os.path.join(
    "data", f"quandl_cpd_{(lbw if lbw else 'none')}lbw"
)

CPD_QUANDL_OUTPUT_FOLDER_DEFAULT = CPD_QUANDL_OUTPUT_FOLDER(CPD_DEFAULT_LBW)

FEATURES_QUANDL_FILE_PATH = lambda lbw: os.path.join(
    "data", f"quandl_cpd_{(lbw if lbw else 'none')}lbw.csv"
)

FEATURES_QUANDL_FILE_PATH_DEFAULT = FEATURES_QUANDL_FILE_PATH(CPD_DEFAULT_LBW)

QUANDL_TICKERS = ["AAPL"]
