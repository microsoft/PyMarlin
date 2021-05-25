(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6695],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return d},kt:function(){return f}});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=a.createContext({}),c=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},d=function(e){var t=c(e.components);return a.createElement(s.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,s=e.parentName,d=l(e,["components","mdxType","originalType","parentName"]),u=c(n),f=i,m=u["".concat(s,".").concat(f)]||u[f]||p[f]||o;return n?a.createElement(m,r(r({ref:t},d),{},{components:n})):a.createElement(m,r({ref:t},d))}));function f(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=u;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:i,r[1]=l;for(var c=2;c<o;c++)r[c]=n[c];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},3159:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return r},metadata:function(){return l},toc:function(){return s},default:function(){return d}});var a=n(2122),i=n(9756),o=(n(7294),n(3905)),r={},l={unversionedId:"plugins/hf_seq_classification",id:"plugins/hf_seq_classification",isDocsHomePage:!1,title:"Text Sequence Classification with Huggingface models",description:"You can use pymarlin.plugins.hfseqclassification for out-of-the-box training of Huggingface models on a downstream sequence classification task. The plugin comes with a golden config file (YAML based). You can simply modify a few arguments for your dataset and you're ready to go.",source:"@site/docs/plugins/hf_seq_classification.md",sourceDirName:"plugins",slug:"/plugins/hf_seq_classification",permalink:"/docs/plugins/hf_seq_classification",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/plugins/hf_seq_classification.md",version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"Named Entity Recognition with HuggingFace models",permalink:"/docs/plugins/hf_ner"},next:{title:"Stats and tensorboard",permalink:"/docs/utils/stats"}},s=[{value:"Walk-thru with a Kaggle dataset",id:"walk-thru-with-a-kaggle-dataset",children:[{value:"Set up the YAML config file",id:"set-up-the-yaml-config-file",children:[]},{value:"Training",id:"training",children:[]},{value:"Evaluate the finetuned model on the test set",id:"evaluate-the-finetuned-model-on-the-test-set",children:[]}]},{value:"Knowledge distillation",id:"knowledge-distillation",children:[]}],c={toc:s};function d(e){var t=e.components,r=(0,i.Z)(e,["components"]);return(0,o.kt)("wrapper",(0,a.Z)({},c,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"You can use ",(0,o.kt)("inlineCode",{parentName:"p"},"pymarlin.plugins.hf_seq_classification")," for out-of-the-box training of Huggingface models on a downstream sequence classification task. The plugin comes with a golden config file (YAML based). You can simply modify a few arguments for your dataset and you're ready to go."),(0,o.kt)("h2",{id:"walk-thru-with-a-kaggle-dataset"},"Walk-thru with a Kaggle dataset"),(0,o.kt)("p",null,"Let us walk through a sample task to better understand the usage. Download the ",(0,o.kt)("a",{parentName:"p",href:"https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv"},"dataset")," for the Coronavirus tweets NLP - Text Classification Kaggle challenge."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"    root   \n    |-- raw_data\n        |-- Corona_NLP_train.csv\n        |-- Corona_NLP_test.csv\n")),(0,o.kt)("p",null,"The plugin uses pandas to read the file into a dataframe, however the expected encoding is utf-8. This dataset has a different encoding so we will need to do some preprocessing. After creating a train-val split, finally save the train, val, test files in separate directories."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'import pandas as pd\nfrom sklearn.model_selection import train_test_split\n\ndf = pd.read_csv("raw_data/Corona_NLP_train.csv", sep=",", encoding=\'latin-1\', header=0)\ndf = df[["OriginalTweet", "Sentiment"]]\ntrain_df, val_df = train_test_split(df, test_size=0.2)\ntrain_df.to_csv("train/train.csv", sep=",", index=False)\nval_df.to_csv("val/val.csv", sep=",", index=False)\ntest_df = pd.read_csv("raw_data/Corona_NLP_test.csv", sep=",", encoding=\'latin-1\', header=0)\ntest_df = test_df[["OriginalTweet", "Sentiment"]]\ntest_df.to_csv("test/test.csv", sep=",", index=False)\n')),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"root\n|-- raw_data\n|   |-- Corona_NLP_train.csv\n|   |-- Corona_NLP_test.csv\n|\n|-- train\n|   |-- train.csv\n|\n|-- val\n|   |-- val.csv\n|\n|-- test\n    |-- test.csv\n")),(0,o.kt)("h3",{id:"set-up-the-yaml-config-file"},"Set up the YAML config file"),(0,o.kt)("p",null,"The dataset contains 2 columns OriginalTweet, Sentiment. The goal is to predict the sentiment of the tweet i.e. text classification with a single sequence. We will try out Huggingface's RoBERTa model for this. For the sake of this tutorial, we will directly use OriginalTweet as the text sequence input to the model with no additional data processing steps or cleanup."),(0,o.kt)("p",null,"Copy the ",(0,o.kt)("inlineCode",{parentName:"p"},"config.yaml")," file from ",(0,o.kt)("a",{target:"_blank",href:n(3571).Z},"here")," to your working directory. You can choose to either edit the config file directly, or override the arguments from commandline. Below is the list of arguments that you need to modify for this dataset."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'# data related args\ndata:\n    train_dir: "./train" # provide path to train dir\n    val_dir: "./test" # provide path to val dir\n    file_format: "csv"\n    header: 0 # file has a header at row 0\n    text_a_col: "OriginalTweet"\n    text_b_col: null # null in config file is equivalent to None\n    label_col: "Sentiment"\n    labels_list: ["Extremely Negative","Negative","Neutral","Positive","Extremely Positive"] # list of labels which will be mapped in order from 0 to 4 for the model\n    hf_tokenizer: "roberta-base" # Huggingface tokenizer name\n\n# model related args\nmodel:\n    hf_model: "roberta-base" # Huggingface model name\n    encoder_key: "roberta" # model key which contains the state dict for RobertaModel\n\n# pymarlin module args\nmodule:\n    metric: "acc_and_f1"\n    max_lr: 0.00001\n\n# trainer args\ntrainer:\n    backend: "sp" # options: sp, sp-amp, ddp, ddp-amp\n    # when running on AML compute, change to ddp or ddp-amp (for mixed precision)\n    epochs: 3\n')),(0,o.kt)("p",null,"You can also override the default values in the config through CLI. For example:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'    $ python run.py --data.train_dir "./train" --data.val_dir "./val" --module.max_lr 0.00005\n')),(0,o.kt)("h3",{id:"training"},"Training"),(0,o.kt)("p",null,"Create a python script (say run.py) with the following lines of code. Alternatively, you can run it through Python interpreter."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.plugins import HfSeqClassificationPlugin\n\nplugin = HfSeqClassificationPlugin()\nplugin.setup()\nplugin.trainer.train()\n")),(0,o.kt)("p",null,"This experiment may be too slow to run on local machine (without gpu). You can switch between trainer backends: sp (singleprocess), sp-amp, ddp, ddp-amp (ddp with mixed precision)."),(0,o.kt)("p",null,"Command to run using DistributedDataParallel:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},"    $ python -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE run.py --config_path='./config.yaml' --trainer.backend \"ddp\"\n")),(0,o.kt)("p",null,"A ",(0,o.kt)("inlineCode",{parentName:"p"},"logs")," folder should have been created which contains tensorboard log."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre"},'    $ tensorboard --logdir="logs"\n')),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Training loss curve",src:n(7445).Z})),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Learning rate",src:n(2624).Z})),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Validation metrics",src:n(8023).Z})),(0,o.kt)("h3",{id:"evaluate-the-finetuned-model-on-the-test-set"},"Evaluate the finetuned model on the test set"),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"config.yaml")," file has a section ",(0,o.kt)("inlineCode",{parentName:"p"},"ckpt")," which contains all checkpointer related arguments. The path specified in ",(0,o.kt)("inlineCode",{parentName:"p"},"model_state_save_dir")," should contain your Pytorch model checkpoints. "),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"# Checkpointer arguments\nckpt:\n    checkpoint: True # Flag indicating whether to checkpoint model.\n    period: 1 # Period of epochs at which to checkpoint model.\n    model_state_save_dir: 'model_ckpts'\n    file_prefix: 'pymarlin' # Prefix of the checkpoint filename.\n    file_ext: 'bin' # File extension for the checkpoint.\n")),(0,o.kt)("p",null,"The model state dict will contain the prefix ",(0,o.kt)("inlineCode",{parentName:"p"},"model")," to all the keys of the Huggingface model state dict. This is because the ",(0,o.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModuleInterface")," holds the Huggingface model in the variable ",(0,o.kt)("inlineCode",{parentName:"p"},"model")," and the pymarling module_interface itself is a torch.nn.Module class."),(0,o.kt)("p",null,"First we will modify the state dict to remove the extra ",(0,o.kt)("inlineCode",{parentName:"p"},"model")," prefix so it can be re-loaded into Huggingface Roberta."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import torch\nfrom collections import OrderedDict\nstate_dict = torch.load('model_ckpts/pymarlin_model_5.bin', map_location='cpu')\nnew_dict = OrderedDict((key.replace('model.',''), value) for key, value in state_dict.items() if key.startswith('model.') )\ntorch.save(new_dict, 'model_ckpts/pymarlin_model_4.bin')\n")),(0,o.kt)("p",null,"Next, we need to edit the ",(0,o.kt)("inlineCode",{parentName:"p"},"config.yaml")," file to point to this model file and the test dataset."),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'# data related args\ndata:\n    train_dir: null # null in config is equivalent to None\n    val_dir: "./test" # provide path to test dataset\n\n# model related args\nmodel:\n    model_path: "model_ckpts"\n    model_file: "pymarlin_model_4.bin"\n    hf_model: "roberta-base" # Huggingface model name\n    encoder_key: "roberta" # model key which contains the state dict for RobertaModel\n')),(0,o.kt)("p",null,"Run the following lines of code to evaluate the finetuned model and compute accuracy and f1 on the test set:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.plugins import HfSeqClassificationPlugin\n\nplugin = HfSeqClassificationPlugin()\nplugin.setup()\nplugin.trainer.validate()\n")),(0,o.kt)("h2",{id:"knowledge-distillation"},"Knowledge distillation"),(0,o.kt)("p",null,"Additionally, you can also distill the finetuned 12 layer Roberta model to a Roberta student, or even a different Huggingface transformer architecture. Only reduction in depth is supported for the plugin. The plugin offers a few loss types: soft labels (logits), hard labels, representations loss (attentions, hidden states)."))}d.isMDXComponent=!0},3571:function(e,t,n){"use strict";t.Z=n.p+"assets/files/config-762f8a6e665172afa1442ed4ba9ac51e.yaml"},7445:function(e,t,n){"use strict";t.Z=n.p+"assets/images/loss-5c6e021ffa6ee945b42e747f8244378d.jpg"},2624:function(e,t,n){"use strict";t.Z=n.p+"assets/images/lr-3540cd6499b7a1227194cb6c30fcb1ba.jpg"},8023:function(e,t,n){"use strict";t.Z=n.p+"assets/images/train_metrics-529f66818a31f2e7c8c071d028e79d2f.jpg"}}]);