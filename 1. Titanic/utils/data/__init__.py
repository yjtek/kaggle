from dataclasses import dataclass
from ..baseclass import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

@dataclass
class TitanicDataset(Dataset):
    
    df: pd.DataFrame
    outcome_col: str
    retain_outcome_col: bool = False
    test_size: float = 0.3
    random_state: int = 123

    def __post_init__(self):
        if self.outcome_col in self.df.columns:
            if not self.retain_outcome_col:
                self.X = (
                    self.df.drop(self.outcome_col)
                ).to_pandas()
                self.y = (
                    self.df.select(self.outcome_col)
                ).to_pandas()
                self.split_train_test()
            else:
                logger.warning(f'Retaining outcome column `{self.outcome_col}` in X. You **must** drop this before training')
                self.X = (
                    self.df
                ).to_pandas()
                self.y = (
                    self.df.select(self.outcome_col)
                ).to_pandas()
                self.split_train_test()

        else:
            self.X = self.df.to_pandas()
            self.y = None

    def split_train_test(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
