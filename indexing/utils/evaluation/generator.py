import sys
import json

from utils.evaluation.dump_eval_result import dump_eval_result

sys.path.append("../..")
import alive_progress
from typing import List, Any, Callable
import time
import pandas as pd
import numpy as np

from algos.algo_interface import IAlgo


class Generator:

    def __init__(
            self,
            questions: List[str],
            embedded_questions: List[Any],
            top_k: int = 10):
        """
        :param questions: Questions to query each runner
        :param top_k: Number of top similar chunks
        """
        self.questions = questions
        self.embedded_questions = embedded_questions
        self.k = top_k
        self.folder_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    def run(self, runners: List[Callable[[], IAlgo]]) -> None:
        pass

    def run_batch(self, runners: List[Callable[[], IAlgo]]) -> None:
        """
        Run a batch of runners
        :param runners: List of runners
        :return:
        """
        total_queries = len(runners)

        with alive_progress.alive_bar(total_queries) as bar:
            for runner in runners:
                start = time.time_ns()
                print("Creating next runner!")
                algo = runner()
                print("Starting runner")

                res = {
                    "question": [],
                    "top_k": [],
                    "time_taken": []
                }

                result, duration = algo.run(np.array(self.embedded_questions), self.k)

                res["question"].append(self.questions)
                res["top_k"].append(result)
                res["time_taken"].append(duration // 1_000_000) # convert to ms

                bar()

                res_df = pd.DataFrame.from_dict(res)
                res_df['top_k'] = res_df['top_k'].apply(lambda x: json.dumps(x.to_dict()))
                end = time.time_ns()
                duration = end - start
                print(f"time taken for runner: {duration // 1_000_000} ms")
                dump_eval_result(self.folder_time, algo.name(), algo.data_source(), res_df, **algo.details())