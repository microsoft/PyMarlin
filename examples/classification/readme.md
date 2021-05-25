https://github.com/Kaggle/kaggle-api

    pip install kaggle
    New-Item ~\.kaggle\kaggle.json
    notepad C:\Users\<user>\.kaggle\kaggle.json

    kaggle datasets download -d datatattle/covid-19-nlp-text-classification

    pip install -r requirements.txt 

# Instructions

1. Install pymarlin library
    pip install pymarlin
    or
    $env:PYTHONPATH=<pymarlin repo path>
2. Set working directory
        cd pymarlin_scripts
3. Prepare data
        python data.py
4. Train
        python train.py [--trainer.max_train_steps_per_epoch 2]

