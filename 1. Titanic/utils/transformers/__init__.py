from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import numpy as np
import re

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
class TransformColToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, levels_threshold: int = 20, fill_nulls: str = '_NULL_VALUE_', replace_original: bool = True):
        self.levels_threshold = levels_threshold
        self.column_name = column_name
        self.fill_nulls = fill_nulls
        self.replace_original = replace_original
        self.new_column_name = column_name if replace_original else f'{column_name}_categorical'
    
    def fit(self, X, y=None):
        levels = X.select(pl.col(f'{self.column_name}')).unique().to_series().to_list()
        self.levels = sorted([x for x in levels if x is not None])

        if len(self.levels) > self.levels_threshold:
            raise ValueError('Column is too high dimensional')

        self.labeller = {
            key: val for val, key in enumerate(self.levels)
        }
        return self
    
    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                pl.col(f'{self.column_name}')
                .apply(lambda x: self.labeller.get(x, f'{self.fill_nulls}'))
                .fill_nan(self.fill_nulls)
                .fill_null(self.fill_nulls)
                .alias(f'{self.new_column_name}')
                .cast(pl.Utf8)
                .cast(pl.Categorical)
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class TransformStringColToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, levels_threshold: int = 20, fill_nulls: str = -1, replace_original: bool = True):
        self.levels_threshold = levels_threshold
        self.column_name = column_name
        self.fill_nulls = fill_nulls
        self.replace_original = replace_original
        self.new_column_name = column_name if replace_original else f'{column_name}_numeric'
    
    def fit(self, X, y=None):
        levels = X.select(pl.col(f'{self.column_name}')).unique().to_series().to_list()
        self.levels = sorted([x for x in levels if x is not None])

        if len(self.levels) > self.levels_threshold:
            raise ValueError('Column is too high dimensional')

        self.labeller = {
            key: val for val, key in enumerate(self.levels)
        }
        return self
    
    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                pl.col(f'{self.column_name}')
                .apply(lambda x: self.labeller.get(x, f'{self.fill_nulls}'))
                .fill_nan(self.fill_nulls)
                .fill_null(self.fill_nulls)
                .alias(f'{self.new_column_name}')
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class TransformColToBins:
    def __init__(self, column_name: str, bin_count: int = 10, fill_nulls=-1, replace_original: bool = True):
        self.column_name: str = column_name
        self.bin_count: int = bin_count
        self.fill_nulls: str = fill_nulls
        self.replace_original = replace_original
        self.new_column_name = column_name if replace_original else f'{column_name}_binned'
    
    def fit(self, X, y=None):
        self.bin_cuts: list = [X[f'{self.column_name}'].quantile(x) for x in np.linspace(0.1,0.9,self.bin_count-1)] #polars includes -inf and inf automatically
        self.bin_labels: list = ['-inf'] + [str(x) for x in self.bin_cuts]
        self.labeller: dict = {val: key for key, val in enumerate(self.bin_labels)}
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                pl.col(self.column_name)
                .cut(self.bin_cuts, self.bin_labels)
                .apply(lambda x: self.labeller.get(x, self.fill_nulls))
                .fill_nan(self.fill_nulls)
                .fill_null(self.fill_nulls)
                .alias(f'{self.new_column_name}')
                .cast(pl.Utf8)
                .cast(pl.Categorical)
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class AddFamilyUnitID(BaseEstimator, TransformerMixin):
    def __init__(self, family_group: list[str]):
        self.family_group: list[str] = family_group
    
    def fit(self, X, y=None):
        self.rank_on_col = X.columns[0]
        return self
    
    def transform(self, X, y=None):
        
        X_transformed = (
            X
            .with_columns(
                pl.when(
                    pl.col(self.rank_on_col)
                    .rank()
                    .over(self.family_group) == 1
                ).then(
                    pl.lit(1)
                ).otherwise(
                    pl.lit(0)
                ).alias(f"groupid_{'_'.join(self.family_group)}")
            )
            .sort(self.family_group)
            .with_columns(
                pl.cumsum(f"groupid_{'_'.join(self.family_group)}").alias(f"groupid_{'_'.join(self.family_group)}")
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class AddSurvivalRate(BaseEstimator, TransformerMixin):
    def __init__(self, survival_rate_group: list[str]):
        self.survival_rate_group = survival_rate_group

    def fit(self, X, y):
        self.survivalrate_by_group_id = (
            pl.concat([X,y], how='horizontal')
            .groupby(self.survival_rate_group)
            .agg(
                pl.mean('Survived').alias(f'survival_rate_{"_".join(self.survival_rate_group)}')
            )
        )
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .join(self.survivalrate_by_group_id, on=self.survival_rate_group, how='left')
        )
        return X_transformed

@PolarsCompatibleTransformer
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop: list[str], regex_to_drop: str):
        self.cols_to_drop: list[str] = cols_to_drop
        self.regex_to_drop: str = regex_to_drop
    
    def fit(self, X, y=None):
        self.drop_cols_with_regex = [col for col in X.columns if re.search(self.regex_to_drop, col) is not None]
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .drop(list(set(self.cols_to_drop + self.drop_cols_with_regex)))
        )
        return X_transformed
