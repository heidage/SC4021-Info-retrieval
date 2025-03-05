from abc import abstractmethod
from typing import Dict, Union, Literal

import pandas as pd

import sys

class IAlgo:

    @abstractmethod
    def run(self,query:str, k:int) -> pd.DataFrame:
        """
        Method to run to search. This method will be executed, where time and results will be recorded.
        :param query: String to be used in the search
        :param k: Number of records to return
        :return: DataFrame with the results
        """
        raise NotImplementedError
    
    @abstractmethod
    def data_source(self) -> Literal["starbucks"]:
        """
        Method to return the name of the data source (e.g. starbucks)
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def details(self) -> Dict[str, Union[str, int]]:
        """
        Method to return a dictionary that details the hyperparameters of the class.
        This will perform similarly to __repr__, but as a purposefully set method.
        :return: dictionary containing kwargs of hyperparameters
        """
        raise NotImplementedError