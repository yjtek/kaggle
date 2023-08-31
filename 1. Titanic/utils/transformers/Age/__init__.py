from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class CleanAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rf = RandomForestRegressor()
    
    def fit(self, X, y=None):
        X_without_missing = X.filter(pl.col('Age').is_not_null())
        outcome = X_without_missing.select('Age').to_pandas()
        features = X_without_missing.select('SibSp', 'Parch', 'Pclass', 'Fare').to_pandas()
        self.rf.fit(features, outcome)
        return self

    def transform(self, X, y=None):

        age_preds = self.rf.predict(X.select('SibSp', 'Parch', 'Pclass', 'Fare').to_pandas())
        X_transformed = (
            X
            .with_columns(pl.Series(age_preds).alias('AgePred'))
            .with_columns(pl.coalesce(pl.col(['Age', 'AgePred'])).alias('Age'))
        )
        return X_transformed