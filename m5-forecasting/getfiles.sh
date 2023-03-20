kaggle competitions download -c m5-forecasting-accuracy --path ./m5-forecasting/data
kaggle competitions download -c m5-forecasting-uncertainty --path ./m5-forecasting/data
unzip ./m5-forecasting/data/m5-forecasting-accuracy.zip -d ./m5-forecasting/data
mv ./m5-forecasting/data/sample_submission.csv ./m5-forecasting/data/sample_submission_accuracy.csv 
unzip ./m5-forecasting/data/m5-forecasting-uncertainty.zip -d ./m5-forecasting/data
mv ./m5-forecasting/data/sample_submission.csv ./m5-forecasting/data/sample_submission_uncertainty.csv
# rm ./data/m5-forecasting-accuracy.zip