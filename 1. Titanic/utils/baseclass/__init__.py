from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import pandas as pd 

@dataclass
class Dataset(metaclass=ABCMeta):

    df: pd.DataFrame
    outcome_col: str

    @abstractmethod
    def split_train_test(self, proportion_test: float) -> tuple[pd.DataFrame]:
        ...

class Transformer(metaclass = ABCMeta):
    ...