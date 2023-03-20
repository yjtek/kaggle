kaggle competitions download -c m5-forecasting-accuracy --path ./data
kaggle competitions download -c m5-forecasting-uncertainty --path ./data
unzip ./data/m5-forecasting-accuracy.zip -d data
mv ./data/sample_submission.csv ./data/sample_submission_accuracy.csv 
unzip ./data/m5-forecasting-uncertainty.zip -d data
mv ./data/sample_submission.csv ./data/sample_submission_uncertainty.csv 
# rm ./data/m5-forecasting-accuracy.zip