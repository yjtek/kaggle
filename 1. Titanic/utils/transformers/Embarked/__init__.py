from ...decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class CleanEmbarked(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                pl.col("Embarked").fill_null('S').alias('Embarked'),
            )
        )
        return X_transformed