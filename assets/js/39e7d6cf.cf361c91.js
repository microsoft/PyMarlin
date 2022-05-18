(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5801],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return m}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),c=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=c(e.components);return r.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),u=c(n),m=a,f=u["".concat(s,".").concat(m)]||u[m]||d[m]||i;return n?r.createElement(f,o(o({ref:t},p),{},{components:n})):r.createElement(f,o({ref:t},p))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,o=new Array(i);o[0]=u;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,o[1]=l;for(var c=2;c<i;c++)o[c]=n[c];return r.createElement.apply(null,o)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},7706:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},metadata:function(){return l},toc:function(){return s},default:function(){return p}});var r=n(2122),a=n(9756),i=(n(7294),n(3905)),o={},l={unversionedId:"plugins/hf_ner",id:"plugins/hf_ner",isDocsHomePage:!1,title:"Named Entity Recognition with HuggingFace models",description:"We designed this plugin to allow for out-of-the-box training and evaluation of HuggingFace models for NER tasks. We provide a golden config file (config.yaml) which you can adapt to your task. This config will make experimentations easier to schedule and track. All the source code and notebooks to submit jobs can be found here",source:"@site/docs/plugins/hf_ner.md",sourceDirName:"plugins",slug:"/plugins/hf_ner",permalink:"/PyMarlin/docs/plugins/hf_ner",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/plugins/hf_ner.md",version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"CNN/DailyMail Summarization",permalink:"/PyMarlin/docs/examples/summarization"},next:{title:"Text Sequence Classification with Huggingface models",permalink:"/PyMarlin/docs/plugins/hf_seq_classification"}},s=[{value:"Step by step with GermEval dataset",id:"step-by-step-with-germeval-dataset",children:[]},{value:"Dataset format",id:"dataset-format",children:[]},{value:"Golden yaml config",id:"golden-yaml-config",children:[]},{value:"Training",id:"training",children:[]},{value:"Evaluation",id:"evaluation",children:[]}],c={toc:s};function p(e){var t=e.components,o=(0,a.Z)(e,["components"]);return(0,i.kt)("wrapper",(0,r.Z)({},c,o,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,"We designed this plugin to allow for out-of-the-box training and evaluation of HuggingFace models for NER tasks. We provide a golden config file (config.yaml) which you can adapt to your task. This config will make experimentations easier to schedule and track. All the source code and notebooks to submit jobs can be found ",(0,i.kt)("a",{parentName:"p",href:"https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2FMarlin_Scenarios%2Fner%2Ftest_nerPlugin"},"here")),(0,i.kt)("h2",{id:"step-by-step-with-germeval-dataset"},"Step by step with GermEval dataset"),(0,i.kt)("p",null,"We will go through how to adapt any dataset/task for PyMarlin and how to setup the plugin. For this purpose we will use the GermEval dataset - this is a dataset with German Named Entity annotation , with data sampled from German Wikipedia and News Corpora. For more granular information and raw dataset please refer ",(0,i.kt)("a",{parentName:"p",href:"https://sites.google.com/site/germeval2014ner/data"},"here")),(0,i.kt)("p",null,"Following HuggingFace documentation for preliminary data clean up we use their preprocess script to clean up the original dataset. These can be run in jupyter Notebook."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"!wget \"https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py\"\n!grep -v \"^#\" NER-de-train.tsv| cut -f 2,3 | tr '\\t' ' ' > train.txt.tmp\n!grep -v \"^#\" NER-de-dev.tsv| cut -f 2,3 | tr '\\t' ' ' > dev.txt.tmp\n!python preprocess.py train.txt.tmp 'bert-base-multilingual-cased' '128' > train.txt\n!python preprocess.py dev.txt.tmp 'bert-base-multilingual-cased' '128' > dev.txt\n!cat train.txt dev.txt | cut -d \" \" -f 2 | grep -v \"^$\"| sort | uniq > labels.txt\n")),(0,i.kt)("h2",{id:"dataset-format"},"Dataset format"),(0,i.kt)("p",null,"NER plugin expects the input to be a TSV or CSV with 2 columns. A column with the text sentences followed by a column with the labels for the tokens in the sentence. For example: 'Sentence': 'who is harry', 'Slot': 'O O B-contact_name'"),(0,i.kt)("p",null,"For GermEval dataset below we show how to modify to format expected by plugin."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"import csv\ndef txt2tsv(filename, outfile):\n  outfile = open(outfile, \"w\")\n  f = open(filename, \"r\")\n  lines = f.readlines()\n  sentence = []\n  labels = []\n  tsv_writer = csv.writer(outfile, delimiter='\\t')\n  tsv_writer.writerow(['Sentence', 'Slot'])\n  for line in lines:\n    line = line.strip()\n    if line:\n      row = line.split(' ')\n      sentence.append(row[0])\n      labels.append(row[1])\n    else:\n      sent = ' '.join(sentence)\n      lab = ' '.join(labels)\n      tsv_writer.writerow([sent, lab])\n      sentence = []\n      labels = []\n\ntxt2tsv(\"dev.txt\", \"dev.tsv\")\ntxt2tsv(\"train.txt\", \"train.tsv\")\n")),(0,i.kt)("p",null,"The dataset would now look like this:"),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"Dataset",src:n(3364).Z})),(0,i.kt)("h2",{id:"golden-yaml-config"},"Golden yaml config"),(0,i.kt)("p",null,"Marlin leverages yaml files for maintaining experiment parameters. For this German Evaluation dataset we provide a golden config ",(0,i.kt)("inlineCode",{parentName:"p"},"config_germ.yaml"),". "),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'# data_processor args\ndata:\n    train_dir : null\n    val_dir : null\n    labels_list: [B-LOC, B-LOCderiv, B-LOCpart, B-ORG, B-ORGderiv, B-ORGpart, B-OTH, B-OTHderiv,\n        B-OTHpart, B-PER, B-PERderiv, B-PERpart, I-LOC, I-LOCderiv, I-LOCpart, I-ORG, I-ORGderiv,\n        I-ORGpart, I-OTH, I-OTHderiv, I-OTHpart, I-PER, I-PERderiv, I-PERpart, O]  \n    tokenizer: "bert-base-multilingual-cased"\n    file_format: "tsv"\n    has_labels: True\n\n# model arguments\nmodel:\n    model_name: "bert"\n    encoder_key: "bert"\n    hf_model: "bert-base-multilingual-cased"\n    model_file: "pytorch_model.bin"\n    model_config_file: "config.json"\n    model_path: null\n    model_config_path: null\n\n# module_interface arguments\nmodule:\n    output_dir: null\n    max_lr : 0.00003 # Maximum learning rate.\n    warmup_prop: 0.1\n    max_seq_len: 128\n    pad_label_id: -100\n    has_labels: True\n\n# trainer arguments\ntrainer:\n    backend: "sp"\n    train_batch_size: 32 # Training global batch size.\n    val_batch_size: 16 # Validation global batch size.\n    epochs: 25 # Total epochs to run.\n    gpu_batch_size_limit : 8 # Max limit for GPU batch size during training.\n    clip_grads : True # Enable or disable clipping of gradients.\n    use_gpu: True # Enable or disable use of GPU.\n    max_grad_norm: 1.0 # Maximum value for gradient norm.\n    writers: [\'stdout\', \'aml\', \'tensorboard\'] # List of all the writers to use.\n    disable_tqdm: True\n    log_level: "DEBUG"\n')),(0,i.kt)("h2",{id:"training"},"Training"),(0,i.kt)("p",null,"Next we need a orchestrating script to initialize the plugin and start training. Assume the script test.py. It will contain the following."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.plugins import HfNERPlugin\nplugin = HfNERPlugin()\nplugin.setup_trainer()\ntrainer = plugin.trainer\ntrainer.train()\ntrainer.validate()\n")),(0,i.kt)("p",null,"We can now schedule a run locally using CLI , modify to point to the train and validation directory appropriately :"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"python test.py --data.train_filepath ./train_germ/train.tsv --data.val_filepath ./val_germ/dev.tsv --config_path config_germ.yaml\n")),(0,i.kt)("h2",{id:"evaluation"},"Evaluation"),(0,i.kt)("p",null,"We specify the path to store the model checkpoints in our config.yaml"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"# Checkpointer arguments\nckpt:\n    checkpoint: True # Flag indicating whether to checkpoint model.\n    delete_existing_checkpoints: True\n    period: 1 # Period of epochs at which to checkpoint model.\n    save_dir: 'ckpts' # Path to directory where MARLIN checkpoints are to be stored.\n    model_save_dir: 'model_ckpts' # Path to directory where MODEL checkpoints are to be stored\n    file_prefix: 'marlin' # Prefix of the checkpoint filename.\n    file_ext: 'bin' # File extension for the checkpoint.\n")),(0,i.kt)("p",null,"The checkpoint consists of all the different components of your scenario like 'module_interface_state', 'trainer_state' and 'trainer_backend_state'. We need to save the model from within the 'module_interface_state' separately. Follow the below steps included in the notebook 'GermEvalAML.ipynb' ",(0,i.kt)("a",{parentName:"p",href:"https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2FMarlin_Scenarios%2Fner%2Fnotebooks%2FGermEvalAML.ipynb"},"here")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"import torch\nfrom collections import OrderedDict\nstate_dict = torch.load('marlin_0.bin', map_location='cpu')\n\nnew_dict = OrderedDict((key.replace('model.',''), value) for key, value in state_dict['module_interface_state'].items() if key.startswith('model.') )\ntorch.save(new_dict, 'marlin_model.bin')\n")),(0,i.kt)("p",null,"Alternatively you could directly load only the model checkpoint as stored in the path specified by ",(0,i.kt)("inlineCode",{parentName:"p"},"model_save_dir"),". In this case you would have the model weights directly in state_dict without needing to go through module_interface_state"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"import torch\nfrom collections import OrderedDict\nstate_dict = torch.load('marlin_0.bin', map_location='cpu')\n\nnew_dict = OrderedDict((key.replace('model.',''), value) for key, value in state_dict.items() if key.startswith('model.') )  \ntorch.save(new_dict, 'marlin_model.bin')\n")),(0,i.kt)("p",null,"Now you can use the model file stored as marlin_model.bin. You can specify this new model as the checkpoint to start from in config.yaml"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},'# model arguments\nmodel:\n    model_name: "bert"\n    encoder_key: "bert"\n    hf_model: "bert-base-multilingual-cased"\n    model_file: "pytorch_model.bin"\n    model_config_file: "config.json"\n    model_path: "<directory containing ckpt>\n    model_config_path: "marlin_model.bin"\n')),(0,i.kt)("p",null,"You can use NER Plugin to evaluate:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"from marlin.plugins import HfNERPlugin\n\nplugin = HfNERPlugin()\nplugin.setup_trainer()\ntrainer = plugin.trainer\ntrainer.train()\ntrainer.validate()\n")))}p.isMDXComponent=!0},3364:function(e,t,n){"use strict";t.Z=n.p+"assets/images/ner_dataset_mod-0d65e2c633fe9b03129eb61780e1fa6c.png"}}]);