from train import *
from mock import MagicMock

def load_classifier():
    config = CustomArgParser().parse()
    checkpoint_path = 'checkpoints/model_9.pt'
    glue_task = config['glue_task']
    data = MagicMock()
    classifier = recipe_factory(glue_task, data_interface = data, **config['mi'])
    sd = torch.load(checkpoint_path,map_location = 'cpu')['module_interface_state']
    classifier.load_state_dict(sd)
    return classifier

if __name__ == "__main__":
    classifier = load_classifier()


    #RTE
    sentence1 = ['No Weapons of Mass Destruction Found in Iraq Yet.',
                'India is a hot country',
                'Krishan has written this inference example']
    sentence2 = ['Weapons of Mass Destruction Found in Iraq.',
                'It\'s warm in india',
                'Krishan is the author of this example']
    input = classifier.tokenizer(
            text = sentence1,
            text_pair = sentence2,
            max_length=classifier.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    output = classifier.net(classifier.encoder(**input))
    result = torch.argmax(output.logits, dim = -1)
    print(result)
    

