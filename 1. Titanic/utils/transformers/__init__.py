from ..decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import numpy as np
import re
from typing import Any
from loguru import logger
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd

@PolarsCompatibleTransformer
class TransformColToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, levels_threshold: int = 20, fill_nulls: str = '-1', replace_original: bool = True):
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
    def __init__(self, column_name: str, levels_threshold: int = 20, fill_nulls: Any = -1, replace_original: bool = True):
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
    def __init__(self, column_name: str, bin_count: int = 10, fill_nulls=-1, replace_original: bool = True, return_numeric=True):
        self.column_name: str = column_name
        self.bin_count: int = bin_count
        self.fill_nulls: str = fill_nulls
        self.replace_original = replace_original
        self.return_numeric = return_numeric
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
            )
        )

        if self.return_numeric:
            X_transformed = (
                X_transformed
                .with_columns(pl.col(self.new_column_name).cast(pl.Int64).alias(self.new_column_name))
            )
        else:
            X_transformed = (
                X_transformed
                .with_columns(pl.col(self.new_column_name).cast(pl.Utf8).cast(pl.Categorical).alias(self.new_column_name))
            )
        
        return X_transformed

@PolarsCompatibleTransformer
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop: list[str] = [], regex_to_drop: str = '', drop_strings: bool = True):
        self.cols_to_drop: list[str] = cols_to_drop
        self.regex_to_drop: str = regex_to_drop
        self.drop_strings: bool = drop_strings
    
    def fit(self, X, y=None):
        self.drop_cols_with_regex = [col for col in X.columns if re.search(self.regex_to_drop, col) is not None] if self.regex_to_drop != '' else []
        
        self.drop_cols_with_string = [colname for colname, coltype in zip(X.columns, X.dtypes) if coltype == pl.Utf8] if self.drop_strings else []
        # self.drop_cols_with_string = [key for key,value in dict(X.dtypes).items() if value == np.dtypes.ObjectDType]
        logger.info(f'String drop cols: {self.drop_cols_with_string}')

        self.final_drop_cols = list(set([colname for colname in self.cols_to_drop + self.drop_cols_with_regex + self.drop_cols_with_string if colname in X.columns]))

        logger.info(f'Specified drop cols: {self.cols_to_drop}')
        logger.info(f'Regex drop cols: {self.drop_cols_with_regex}')
        logger.info(f'String drop cols: {self.drop_cols_with_string}')
        logger.info(f'Dropping cols: {self.final_drop_cols}')

        return self

    def transform(self, X, y=None):
        only_drop_present_cols = [col for col in X.columns if col in self.final_drop_cols]
        X_transformed = (
            X
            .drop(*only_drop_present_cols)
        )
        return X_transformed

@PolarsCompatibleTransformer
class FillNull(BaseEstimator, TransformerMixin):
    def __init__(self, null_value: float):
        self.null_value = null_value
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .fill_null(self.null_value)
            .fill_nan(self.null_value)
        )
        return X_transformed

@PolarsCompatibleTransformer
class AddRandomColumn(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.random_state = 123
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        np.random.seed(self.random_state)
        X_transformed = (
            X
            .with_columns(random = pl.lit(np.random.rand(X.height)))
        )
        return X_transformed

class PipelineCompatibleCatBoostClassifier(CatBoostClassifier):
    # def __init__(self, **kwargs):
    #     ...

    def __init__(self, random_seed: int, **kwargs):
        super().__init__(verbose=False, random_seed=random_seed, **kwargs)

    def fit(self, X, y=None, **kwargs):
        self.cat_features = [colname for colname, coltype in dict(X.dtypes).items() if type(coltype) == pd.core.dtypes.dtypes.CategoricalDtype]
        model = (
            super()
            .fit(X, y=y, cat_features=self.cat_features, **kwargs)
        )
        return model

@PolarsCompatibleTransformer
class StandardiseDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
        
    def fit(self, X, y=None):
        self.numeric_cols = (
            X
            .select(pl.exclude([pl.Categorical, pl.Utf8]))
            .drop('Survived')
            .columns
        )

        self.means = (
            X
            .select(self.numeric_cols)
            .groupby(pl.lit(1).alias('dropcol'))
            .agg(pl.mean('*'))
            .drop('dropcol')
        )
        self.means = {key:value for key,value in zip(self.means.columns, self.means.row(0))}

        self.stds = (
            X
            .select(self.numeric_cols)
            .groupby(pl.lit(1).alias('dropcol'))
            .agg(pl.std('*'))
            .drop('dropcol')
        )
        self.stds = {key:value for key,value in zip(self.stds.columns, self.stds.row(0))}
        return self

    def transform(self, X, y=None):
        
        X_transformed = (
            X
            .select(self.numeric_cols)
            .with_columns(*[
                (pl.col(x) - self.means.get(x) / self.stds.get(x)).alias(x) 
                for x in self.numeric_cols
            ])
            .hstack(
                X.drop(self.numeric_cols)
            )
            .select(X.columns)
        )
        return X_transformed

@PolarsCompatibleTransformer
class ReduceDimIsoMap(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, **kwargs):
        self.reducer = Isomap(n_components=n_components, **kwargs)
        
    def fit(self, X, y=None):
        self.reducer.fit(X)
        return self

    def transform(self, X, y=None):
        X_transformed = pl.DataFrame(self.reducer.transform(X))
        return X_transformed


@PolarsCompatibleTransformer
class ReduceDimPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.pca = PCA(n_components = n_components)

    def fit(self, X, y=None):
        self.pca.fit(X.to_pandas())
        return self

    def transform(self, X, y=None):
        X_transformed = pl.DataFrame(
            self.pca.transform(X)
        )
        return X_transformed

@PolarsCompatibleTransformer
class ReduceDimTruncSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.tsvd = TruncatedSVD(n_components = n_components)

    def fit(self, X, y=None):
        self.tsvd.fit(X)
        return self

    def transform(self, X, y=None):
        X_transformed = pl.DataFrame(
            self.tsvd.transform(X)
        )
        return X_transformed

from sklearn.feature_selection import SelectKBest, mutual_info_classif

@PolarsCompatibleTransformer
class SelectKBestFeats(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=20):
        self.skb = SelectKBest(mutual_info_classif, k=n_features)

    def fit(self, X, y=None):
        X_numeric_only = (
            X
            .select(pl.exclude([pl.Utf8, pl.Categorical]))
            .drop('Survived')
            .to_pandas()
        )
        self.skb.fit(X_numeric_only, np.array(X['Survived']).astype('int'))
        self.non_numeric_cols = list(X.select(pl.col([pl.Utf8, pl.Categorical])).columns)
        return self

    def transform(self, X, y=None):
        
        if ('Survived' in X.columns):
            X_transformed = (
                X
                .select(list(self.skb.get_feature_names_out()) + self.non_numeric_cols + ['Survived'])
            )
        else:
            X_transformed = (
                X
                .select(list(self.skb.get_feature_names_out()) + self.non_numeric_cols)
            )
        
        return X_transformed