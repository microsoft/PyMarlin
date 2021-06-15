from pymarlin.core.data_interface import DataInterface, DataProcessor
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

cache_dir  =r"/tmp/hf_data"
class GlueDataAnalyzer(DataProcessor):
    def __init__(self, glue_task):
        self.datasets = load_dataset("glue",glue_task, cache_dir = cache_dir)
    
    def process(self):
        pass

    def analyze(self):
        print(self.datasets)
        for split in self.datasets.keys():
            self.analyze_split(split)
        
    def analyze_split(self,split = 'train'):
        print(f'\n{split} data label distribution')
        df = pd.DataFrame(self.datasets[split])
        print(df.head(2))
        count = df.groupby('label')['label'].count()
        summary = pd.DataFrame({'count':count, 'ratio': count/len(df)})
        print(summary)
        return df
    
    def analyze_texts(self, texts):
        s = pd.Series( texts)
        s = s.apply(lambda cell : cell.split())
        print(s.apply(len).describe(percentiles = [0.5,0.95,0.99,0.999]))

class SentenceDataAnalyzer(GlueDataAnalyzer):
    def __init__(self, glue_task, sentence_key = 'sentence'):
        super().__init__(glue_task)
        self.sentence_key = sentence_key
    def analyze_split(self, split = 'train'):
        df = super().analyze_split(split)
        self.analyze_texts(df[self.sentence_key])

class SPDataAnalyzer(GlueDataAnalyzer):
    def __init__(self, glue_task, s1_key = 'question1', s2_key = 'question2'):
        super().__init__(glue_task)
        self.s1_key = s1_key
        self.s2_key = s2_key
    def analyze_split(self, split = 'train'):
        df = super().analyze_split(split)
        self.analyze_texts(df[self.s1_key])
        self.analyze_texts(df[self.s2_key])


class SPRegressionDataAnalyzer(SPDataAnalyzer):
    def __init__(self, glue_task, s1_key = 'question1', s2_key = 'question2'):
        super().__init__(glue_task)
        self.s1_key = s1_key
        self.s2_key = s2_key
    def analyze_split(self, split = 'train'):
        print(f'\n{split} data label distribution')
        df = pd.DataFrame(self.datasets[split])
        print(df.head(2))
        # print(df.label.describe())
        self.analyze_texts(df[self.s1_key])
        self.analyze_texts(df[self.s2_key])

def analyzer_factory(glue_task):
    factory = {
        'default':GlueDataAnalyzer(glue_task),
        'qqp':SPDataAnalyzer('qqp'),
        'rte':SPDataAnalyzer('rte', 'sentence1', 'sentence2'),
        'mnli':SPDataAnalyzer('mnli', 'premise', 'hypothesis'),
        'qnli':SPDataAnalyzer('qnli', 'question', 'sentence'),
        'sst2':SentenceDataAnalyzer('sst2'),
        'stsb':SPRegressionDataAnalyzer('stsb', 'sentence1','sentence2'),
        'wnli':SPDataAnalyzer('wnli', 'sentence1', 'sentence2'),
        'mrpc':SPDataAnalyzer('mrpc', 'sentence1', 'sentence2'),
    }
    glue_task = glue_task if glue_task in factory else 'default'
    return factory[glue_task]

class GlueData(DataInterface):
    def setup_datasets(self, glue_task = 'cola'):
        self.glue_task = glue_task
        datasets = load_dataset("glue",glue_task, cache_dir = cache_dir)
        self.train_ds = datasets['train']
        if glue_task == 'mnli':
            self.val_ds = {'mnli_matched':datasets['validation_matched'],'mnli_mismatched':datasets['validation_mismatched']}
            self.test_ds = [datasets['test_matched'],datasets['test_mismatched']]
        else:
            self.val_ds = datasets['validation']
            self.test_ds = datasets['test']
    
    def get_train_dataset(self):
        return self.train_ds
    
    def get_val_dataset(self):
        return self.val_ds 

    def get_test_dataset(self):
        return self.test_ds

if __name__ == "__main__":
    import sys
     
    glue_task = sys.argv[1] if len(sys.argv) >1 else 'cola'
    print(glue_task)
    di = GlueData(glue_task)
    di.setup_datasets()
    di.process_data(analyzer_factory(glue_task))

#python src/data_hf_glue.py rte