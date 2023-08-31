from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class CleanFare(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        self.median_fare_by_pclass = (
            X
            .groupby('Pclass')
            .agg(pl.col('Fare').median().alias('MedianFare'))
        )
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .join(self.median_fare_by_pclass, on='Pclass', how='left')
            .with_columns(
                pl.coalesce(["Fare", "MedianFare"]).alias("Fare")
            )
        )
        
        return X_transformed