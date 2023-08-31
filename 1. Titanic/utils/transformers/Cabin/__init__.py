from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class CleanCabin(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                pl.col("Cabin").fill_null('').alias('Cabin'),
                pl.col("Cabin").str.slice(0,1).alias('CabinFirstLetter')
            )
        )
        return X_transformed