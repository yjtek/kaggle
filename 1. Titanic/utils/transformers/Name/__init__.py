from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class MakeNameFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_name_transformed = (
            X
            .select(
                '*',
                pl.col('Name').str.split(by=', ').apply(lambda x: x[0]).alias('LastName'),
                pl.col('Name').str.split(by=', ').apply(lambda x: x[1]).str.split(by=' ').apply(lambda x: x[0]).alias('Honorific')
            )
            .drop('Name')
        )
        return X_name_transformed


@PolarsCompatibleTransformer
class CleanHonorific(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            # .with_columns(
            #     pl.when(
            #         pl.col("Honorific").is_in([''])
            #     ).then(

            #     )
            # )
        )
        return X_transformed