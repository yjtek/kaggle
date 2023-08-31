from dataclasses import dataclass
from ..baseclass import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import polars as pl

@dataclass
class TitanicDataset(Dataset):
    
    df: pd.DataFrame
    outcome_col: str
    retain_outcome_col: bool = False
    test_size: float = 0.3
    random_state: int = 123
    predict_only: bool = False

    def __post_init__(self):
        if self.outcome_col in self.df.columns:
            if not self.retain_outcome_col:
                self.X = (
                    self.df.drop(self.outcome_col)
                ).to_pandas()
                self.y = (
                    self.df.select(self.outcome_col)
                ).to_pandas()    
            else:
                logger.warning(f'Retaining outcome column `{self.outcome_col}` in X. You **must** drop this before training')
                self.X = (
                    self.df
                ).to_pandas()
                self.y = (
                    self.df.select(self.outcome_col)
                ).to_pandas()
            
            if not self.predict_only:
                self.split_train_test()

        else:
            self.X = self.df.to_pandas()
            self.y = None

    def split_train_test(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)


def clean_test_to_fit_answer_sheet(df: pl.DataFrame) -> pl.DataFrame:
    map_wrong_to_right_names: dict = {
        '''Assaf Khalil, Mrs. Mariana (Miriam")"''': '''Assaf Khalil, Mrs. Mariana ("Miriam")''',
        '''Johnston, Mrs. Andrew G (Elizabeth Lily" Watson)"''': '''Johnston, Mrs. Andrew G (Elizabeth "Lily" Watson)''',
        '''Katavelas, Mr. Vassilios (Catavelas Vassilios")"''': '''Katavelas, Mr. Vassilios ("Catavelas Vassilios")''',
        '''Coutts, Mrs. William (Winnie Minnie" Treanor)"''': '''Coutts, Mrs. William (Winnie "Minnie" Treanor)''',
        '''Hocking, Miss. Ellen Nellie""''': '''Hocking, Miss. Ellen "Nellie"''',
        '''Thomas, Mrs. Alexander (Thamine Thelma")"''': '''Thomas, Mrs. Alexander (Thamine "Thelma")''',
        '''Willer, Mr. Aaron (Abi Weller")"''': '''Willer, Mr. Aaron ("Abi Weller")''',
        '''Lindeberg-Lind, Mr. Erik Gustaf (Mr Edward Lingrey")"''': '''Lindeberg-Lind, Mr. Erik Gustaf ("Mr Edward Lingrey")''',
        '''Moubarek, Mrs. George (Omine Amenia" Alexander)"''': '''Moubarek, Mrs. George (Omine "Amenia" Alexander)''',
        '''Johnston, Master. William Arthur Willie""''': '''Johnston, Master. William Arthur "Willie"''',
        '''Khalil, Mrs. Betros (Zahie Maria" Elias)"''': '''Khalil, Mrs. Betros (Zahie "Maria" Elias)''',
        '''Wells, Mrs. Arthur Henry (Addie" Dart Trevaskis)"''': '''Wells, Mrs. Arthur Henry ("Addie" Dart Trevaskis)''',
        '''Daly, Miss. Margaret Marcella Maggie""''': '''Daly, Miss. Margaret Marcella "Maggie"''',
        '''McCarthy, Miss. Catherine Katie""''': '''McCarthy, Miss. Catherine "Katie"''',
        '''Rosenshine, Mr. George (Mr George Thorne")"''': '''Rosenshine, Mr. George ("Mr George Thorne")''',
        '''Nakid, Mrs. Said (Waika Mary" Mowad)"''': '''Nakid, Mrs. Said (Waika "Mary" Mowad)''',
        '''Dean, Miss. Elizabeth Gladys Millvina""''': '''Dean, Miss. Elizabeth Gladys "Millvina"''',
        '''Riihivouri, Miss. Susanna Juhantytar Sanni""''': '''Riihivouri, Miss. Susanna Juhantytar "Sanni"''',
        '''Cotterill, Mr. Henry Harry""''': '''Cotterill, Mr. Henry "Harry"''',
        '''Wheeler, Mr. Edwin Frederick""''': '''Wheeler, Mr. Edwin "Frederick"''',
        '''Nourney, Mr. Alfred (Baron von Drachstedt")"''': '''Nourney, Mr. Alfred ("Baron von Drachstedt")''',
        '''Riordan, Miss. Johanna Hannah""''': '''Riordan, Miss. Johanna "Hannah"'''
    }

    df_clean = (
        df
        .with_columns(pl.col('Name').apply(lambda x: map_wrong_to_right_names.get(x)).alias('NameAmended'))
        .with_columns(pl.when(pl.col('NameAmended').is_not_null()).then(pl.col('NameAmended')).otherwise(pl.col("Name")).alias('Name'))
        .drop("NameAmended")
    )

    return df_clean

def submit_answer(df):
    submission = pl.DataFrame({
        'PassengerId': df['PassengerId'],
        # 'Survived': PIPELINE.predict(test_transformed.X),
        # 'Survived': np.where(test_transformed.X['Sex'] == 'female', 1, 0)
        'Survived': df['Survived']
    })
    filename = f'''./submission/submit_{datetime.now().strftime('%Y%m%d_%H%M')}.csv'''
    #display(submission.head())
    submission.write_csv(filename)
    os.system(f'''kaggle competitions submit -c titanic -f {filename} -m {filename}''')