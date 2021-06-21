# Instructions
1. Install requirements
        pip install -r requirements.txt
2. Download data from kaggle
        Ref: https://github.com/Kaggle/kaggle-api
        Get your credentials file from kaggle here: C:\Users\<user>\.kaggle\kaggle.json
        kaggle datasets download -d datatattle/covid-19-nlp-text-classification
3. Install pymarlin library
        pip install pymarlin
        or
        $env:PYTHONPATH=<pymarlin repo path>
4. Set working directory
        cd pymarlin_scripts
5. Prepare data
        python data.py
6. Train
        python train.py [--trainer.max_train_steps_per_epoch 2]