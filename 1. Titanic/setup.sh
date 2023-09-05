source ~/.pyenv/versions/kaggle/bin/activate
kaggle competitions download -c titanic -p ./data
unzip -o ./data/*.zip -d ./data
python3 ./data/setup_test.py