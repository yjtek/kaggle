from ...decorators import PolarsCompatibleTransformer
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
            .with_columns(pl.col("Cabin").fill_null('').alias('Cabin'))
            .with_columns(pl.col("Cabin").str.slice(0,1).alias('CabinFirstLetter'))
            
            .with_columns(pl.when(pl.col('CabinFirstLetter') == pl.lit("T")).then(pl.lit('A')).otherwise(pl.col('CabinFirstLetter')).alias('CabinFirstLetter'))
            .with_columns(
                pl.when(
                    pl.col('CabinFirstLetter').is_in(['A','B','C'])
                )
                .then(
                    pl.lit('ABC')
                ).when(
                    pl.col('CabinFirstLetter').is_in(['D','E'])
                ).then(
                    pl.lit('DE')
                ).when(
                    pl.col('CabinFirstLetter').is_in(['F','G'])
                ).then(
                    pl.lit('FG')
                ).otherwise(
                    pl.lit('M')
                ).alias('CabinFirstLetterGroups')
            )
        )
        
        return X_transformed