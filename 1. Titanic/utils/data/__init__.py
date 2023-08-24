from dataclasses import dataclass
from ..baseclass import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class TitanicDataset(Dataset):
    
    df: pd.DataFrame
    outcome_col: str

    def __post_init__(self):
        if self.outcome_col in self.df.columns:
            self.X = (
                self.df.drop(self.outcome_col)
            ).to_pandas()
            self.y = (
                self.df.select(self.outcome_col)
            ).to_pandas()
            self.split_train_test()
        else:
            self.X = self.df.to_pandas()
            self.y = None

    def split_train_test(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.3)
