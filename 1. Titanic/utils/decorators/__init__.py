import polars as pl
from typing import Callable
from sklearn.base import TransformerMixin

class PolarsCompatibleTransformer:
    
    def __init__(self, cls):
        #super().__init__(*args, **kwargs)
        self.cls = cls

    def polars_compatible(self, func: Callable, functype: str) -> Callable:
        def polars_function(X, y=None, *args, **kwargs):
            X_polars = pl.from_pandas(X)
            
            if y is not None:
                y_polars = pl.from_pandas(y)
                result = func(X_polars, y_polars)
            else:
                result = func(X_polars)
            
            if (result is not None) and (not isinstance(result, TransformerMixin)) and (functype == 'transform'):
                X_pandas = result.to_pandas()
                return X_pandas
            
            return result

        return polars_function
    
    def __call__(self, *args, **kwargs):

        decorated_cls = self.cls(*args, **kwargs)
        decorated_cls.fit = self.polars_compatible(decorated_cls.fit, functype = 'fit')
        decorated_cls.transform = self.polars_compatible(decorated_cls.transform, functype = 'transform')
        return decorated_cls