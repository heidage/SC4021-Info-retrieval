import os
import sys
import time
import alive_progress

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from typing import List, Tuple
from utils.load_data import load_csv, load_parquet
from utils.evaluation.dump_eval_result import dump_eval_result
from algos.VSMAlgo import VSMAlgo