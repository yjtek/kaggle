from ...decorators import PolarsCompatibleTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl

@PolarsCompatibleTransformer
class MakeIsWomanOrBoyIndicator(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                IsWomanOrBoy = (
                    pl.when(
                        (pl.col('Sex') == 'female') | (pl.col('Honorific') == 'Master')
                    ).then(
                        1
                    ).otherwise(
                        0
                    )
                )
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class MakeFamilySurvivedRate(BaseEstimator, TransformerMixin):
    def __init__(self, groups: list[str]):
        self.groups = groups
        self.group_namestring = '_'.join(self.groups)

    def fit(self, X, y=None):
        self.family_group_agg = (
            X
            .groupby(self.groups)
            .agg(
                pl.col('PassengerId').n_unique().alias(f'FamilyCount_{self.group_namestring}'),

                pl.col('Survived').sum().alias(f'FamilySurvivedCount_{self.group_namestring}'),

                pl.col('IsWomanOrBoy').sum().alias(f'FamilyWomanOrBoyCount_{self.group_namestring}'),

                pl.when(
                    pl.col('IsWomanOrBoy') == 1
                ).then(
                    pl.col('Survived')
                ).otherwise(
                    0
                ).sum().alias(f'FamilyWomanOrBoySurvivedCount_{self.group_namestring}'),
            )
            # .with_columns(
            #     pl.when(
            #         pl.col(f'FamilyCount_{self.group_namestring}') == 1
            #     ).then(
            #         1
            #     ).otherwise(
            #         0
            #     ).alias(f'FamilyCount_IsSingle')
            # )
            .filter(
                pl.col(f'FamilyCount_{self.group_namestring}') > 1
            )
        )
        return self

    def transform(self, X, y=None):
        X_transformed = (
            X
            .join(self.family_group_agg, on=self.groups, how='left')
            .with_columns(
                pl.when(
                    pl.col('IsWomanOrBoy') == 1
                ).then(
                    pl.col(f'FamilyWomanOrBoyCount_{self.group_namestring}') - 1
                ).otherwise(
                    pl.col(f'FamilyWomanOrBoyCount_{self.group_namestring}')
                )
            )
            .with_columns(
                (pl.col(f'FamilyWomanOrBoySurvivedCount_{self.group_namestring}')/pl.col(f'FamilyWomanOrBoyCount_{self.group_namestring}'))
                .alias(f'FamilyWomanOrBoySurvivedRate_{self.group_namestring}'),

                (pl.col(f'FamilySurvivedCount_{self.group_namestring}')/pl.col(f'FamilyCount_{self.group_namestring}'))
                .alias(f'FamilySurvivedRate_{self.group_namestring}')
            )
            .fill_nan(0)
        )
        return X_transformed

@PolarsCompatibleTransformer
class AddFamilySize(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                FamilySize = pl.col('SibSp') + pl.col('Parch') + 1)
            .with_columns(
                FamilySizeGroup = pl.when(
                    pl.col('FamilySize') == 1
                ).then(
                    1
                ).when(
                    pl.col('FamilySize').is_between(2,4, closed='both')
                ).then(
                    2
                ).when(
                    pl.col('FamilySize').is_between(5,6, closed='both')
                ).then(
                    3
                ).otherwise(
                    4
                ).alias('FamilySizeGroup')
            )
        )
        return X_transformed

@PolarsCompatibleTransformer
class AddIsMarried(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = (
            X
            .with_columns(
                IsMarried = pl.when(pl.col('Honorific') == 'Mrs').then(1).otherwise(0)
            )
        )
        return X_transformed