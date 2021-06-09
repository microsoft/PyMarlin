'''
Inferencing after training completion
'''
import torch
import os
import argparse

from transformers import BartForConditionalGeneration, BartTokenizerFast
class Summarizer(torch.nn.Module):
    def __init__(self, model_path = 'outputs', model_file='', isCheckpoint = True, load_weights = True):
        super().__init__()
        self.fullpath = os.path.join(model_path, model_file)
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #load model weights
        if load_weights:
            self._load_weights(isCheckpoint)
        
    
    def _load_weights(self, isCheckpoint = True):
        state_dict = torch.load(self.fullpath)
        if isCheckpoint:
            state_dict = state_dict['module_interface']
            self.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def summarize(self, text):
        batch = self.tokenizer(text, return_tensors='pt').to(self.device)
        generated_ids = self.model.generate(batch['input_ids'])
        return self.tokenizer.batch_decode(generated_ids)[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=r"checkpoints", help="Path to model")
    parser.add_argument("--model_file", type=str, default = "model_0.tar")

    args = parser.parse_args()
    summ = Summarizer(model_path = args.model_path, model_file = args.model_file )
    text = "Home Secretary Priti Patel warns people trying to leave UK will be turned back at airports and lashes influencers 'working' in the sun as she unveils quarantine rules for Brits returning from 30 high-risk countries"
    summary = summ.summarize(text)
    print(text)
    print('Summary:', summary)