from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / 'data_raw'
DATA_CANONICAL = PROJECT_ROOT / 'data_canonical'
DATA_FORMATTED = PROJECT_ROOT / 'data_formatted'
DATA_FOR_TABLEAU = PROJECT_ROOT / 'data_for_tableau'

TARGET_COL = 'job_category'  # change if needed
RANDOM_STATE = 42
