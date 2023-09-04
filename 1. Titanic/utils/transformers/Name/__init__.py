from ...decorators import PolarsCompatibleTransformer
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
                pl.col('Name').str.split(by=', ').apply(lambda x: x[1]).str.split(by='. ').apply(lambda x: x[0]).alias('Honorific')
            )
            .drop('Name')
        )
        return X_name_transformed


@PolarsCompatibleTransformer
class CleanHonorific(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.honorific_map: dict = {
            'woman': ['Miss', 'Mrs', 'Lady', 'Mlle', 'the Countess'],
            'boy': ['Master'],
            'man': [True]
        }
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                HonorificGrouped = (pl.when(
                    pl.col("Honorific").is_in(['Miss', 'Mrs', 'Lady', 'Mlle', 'the Countess'])
                ).then(
                    'woman'
                ).when(
                    pl.col('Honorific').is_in(['Master'])
                ).then(
                    'boy'
                ).otherwise(
                    'man'
                ))
            )
        )
        #display(X_transformed.head())
        return X_transformed

@PolarsCompatibleTransformer
class CleanLastName(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self, X, y=None):
        self.lastname_freq = (
            X
            .groupby('LastName')
            .agg(
                LastNameFreq = pl.col('PassengerId').count()
            )
        )
        
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .join(self.lastname_freq, on='LastName', how='left')
            .with_columns(
                LastNameClean = (pl.when(
                    pl.col("HonorificGrouped") == 'man'
                ).then(
                    'noGroup'
                ).otherwise(
                    pl.col('LastName')
                ))
            )
            .with_columns(
                LastNameClean = (pl.when(
                    pl.col('LastNameFreq') <= 1
                ).then(
                    'noGroup'
                ).otherwise(
                    pl.col('LastName')
                ))
            )
        )
        #display(X_transformed.head())
        return X_transformed

