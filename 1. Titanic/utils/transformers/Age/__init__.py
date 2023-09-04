from ...decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
from sklearn.ensemble import RandomForestRegressor

@PolarsCompatibleTransformer
class CleanAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rf = RandomForestRegressor(random_state=123)
    
    def fit(self, X, y=None):
        X_without_missing = X.filter(pl.col('Age').is_not_null())
        outcome = X_without_missing.select('Age').to_pandas()
        features = (
            X_without_missing
            .with_columns(SexDummy = pl.when(pl.col('Sex') == 'male').then(1).otherwise(0)).select('SibSp', 'Parch', 'Pclass', 'Fare', 'SexDummy')
            .to_pandas()
        )
        self.rf.fit(features, outcome)
        return self

    def transform(self, X, y=None):

        age_preds = self.rf.predict(
            X
            .with_columns(SexDummy = pl.when(pl.col('Sex') == 'male').then(1).otherwise(0))
            .select('SibSp', 'Parch', 'Pclass', 'Fare', 'SexDummy').to_pandas()
        )
        X_transformed = (
            X
            .with_columns(pl.Series(age_preds).alias('AgePred'))
            .with_columns(pl.coalesce(pl.col(['Age', 'AgePred'])).alias('Age'))
        )
        return X_transformed