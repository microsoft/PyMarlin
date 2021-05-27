(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9840],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return c},kt:function(){return f}});var a=t(7294);function r(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function i(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?i(Object(t),!0).forEach((function(n){r(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,a,r=function(e,n){if(null==e)return{};var t,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||(r[t]=e[t]);return r}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var s=a.createContext({}),p=function(e){var n=a.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},c=function(e){var n=p(e.components);return a.createElement(s.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},d=a.forwardRef((function(e,n){var t=e.components,r=e.mdxType,i=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),d=p(t),f=r,m=d["".concat(s,".").concat(f)]||d[f]||u[f]||i;return t?a.createElement(m,o(o({ref:n},c),{},{components:t})):a.createElement(m,o({ref:n},c))}));function f(e,n){var t=arguments,r=n&&n.mdxType;if("string"==typeof e||r){var i=t.length,o=new Array(i);o[0]=d;var l={};for(var s in n)hasOwnProperty.call(n,s)&&(l[s]=n[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,o[1]=l;for(var p=2;p<i;p++)o[p]=t[p];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}d.displayName="MDXCreateElement"},2862:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return o},metadata:function(){return l},toc:function(){return s},default:function(){return c}});var a=t(2122),r=t(9756),i=(t(7294),t(3905)),o={sidebar_label:"implementation",title:"plugins.hf_ner.implementation"},l={unversionedId:"reference/plugins/hf_ner/implementation",id:"reference/plugins/hf_ner/implementation",isDocsHomePage:!1,title:"plugins.hf_ner.implementation",description:"HfNERPlugin Objects",source:"@site/docs/reference/plugins/hf_ner/implementation.md",sourceDirName:"reference/plugins/hf_ner",slug:"/reference/plugins/hf_ner/implementation",permalink:"/docs/reference/plugins/hf_ner/implementation",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/plugins/hf_ner/implementation.md",version:"current",sidebar_label:"implementation",frontMatter:{sidebar_label:"implementation",title:"plugins.hf_ner.implementation"},sidebar:"referenceSideBar",previous:{title:"plugins.hf_ner.data_classes",permalink:"/docs/reference/plugins/hf_ner/data_classes"},next:{title:"plugins.hf_ner.module_classes",permalink:"/docs/reference/plugins/hf_ner/module_classes"}},s=[{value:"HfNERPlugin Objects",id:"hfnerplugin-objects",children:[]}],p={toc:s};function c(e){var n=e.components,t=(0,r.Z)(e,["components"]);return(0,i.kt)("wrapper",(0,a.Z)({},p,t,{components:n,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"hfnerplugin-objects"},"HfNERPlugin Objects"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"class HfNERPlugin(Plugin)\n")),(0,i.kt)("p",null,"Named Entity Recognition or Token Classification plugin for HuggingFace models"),(0,i.kt)("p",null,"plugin.setup() bootstraps the entire pipeline and returns a fully setup trainer."),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Example"),":"),(0,i.kt)("p",null,"  trainer = plugin.setup()\ntrainer.train()\ntrainer.validate()"),(0,i.kt)("p",null,"  Alternatively, you can run ",(0,i.kt)("inlineCode",{parentName:"p"},"setup_datainterface")," ",(0,i.kt)("inlineCode",{parentName:"p"},"setup_module")," ",(0,i.kt)("inlineCode",{parentName:"p"},"setup_trainer")," individually."),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Example"),":"),(0,i.kt)("p",null,"  plugin.setup_datainterface()\nplugin.setup_module()\ntrainer = plugin.setup_trainer()"),(0,i.kt)("h4",{id:"__init__"},"_","_","init","_","_"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"}," | __init__(config: Optional[Dict] = None)\n")),(0,i.kt)("p",null,"CustomArgParser parses YAML config located at cmdline --config_path. If --config_path\nis not provided, assumes YAML file is named config.yaml and present in working directory.\nInstantiates dataclasses:\nself.data_args (arguments.DataInterfaceArguments): Instantiated dataclass containing\nargs required to initialize NERDataInterface and NERProcessor classes\nself.module_args (arguments.ModuleInterfaceArguments): Instantiated dataclass containing\nargs required to initialize NERModule class"),(0,i.kt)("p",null,"Sets properties:\nself.datainterface: data_interface.DataInterface ","[NERDataInterface]"," object\nself.dataprocessor: data_interface.DataProcessor ","[NERProcessor]"," object.\nThese two together are used to read raw data and create sequences of tokens in ",(0,i.kt)("inlineCode",{parentName:"p"},"setup_datainterface"),".\nThe processed data is fed to HuggingFace AutoModelForTokenClassification models.\nself.module: module_interface.ModuleInterface ","[NERModule]"," object\nThis is used to initialize a Marlin trainer."),(0,i.kt)("h4",{id:"setup_datainterface"},"setup","_","datainterface"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"}," | setup_datainterface()\n")),(0,i.kt)("p",null,"Executes the data processing pipeline. Tokenizes train and val datasets using the\n",(0,i.kt)("inlineCode",{parentName:"p"},"dataprocessor")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"datainterface"),".\nFinally calls ",(0,i.kt)("inlineCode",{parentName:"p"},"datainterface.setup_datasets(train_data, val_data)"),"."),(0,i.kt)("p",null,"Assumptions:\nTraining and validation files are placed in separate directories.\nAccepted file formats: tsv, csv.\nFormat of input files 2 columns ","'","Sentence","'",", ","'","Slot","'","\nExample\n{","'","Sentence","'",": ","'","who is harry","'",",\n","'","Slot","'",": ","'","O O B-contact_name","'","},"),(0,i.kt)("h4",{id:"setup_module"},"setup","_","module"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"}," | setup_module(module_interface=None)\n")),(0,i.kt)("p",null,"Sets ",(0,i.kt)("inlineCode",{parentName:"p"},"NERModule.data")," property to ",(0,i.kt)("inlineCode",{parentName:"p"},"datainterface")," which contains\nthe processed datasets. Assertion error is thrown if ",(0,i.kt)("inlineCode",{parentName:"p"},"datainterface")," retrieves no train\nor val data, indicating that ",(0,i.kt)("inlineCode",{parentName:"p"},"datainterface")," hasn","'","t been setup with processed data.\nSets the ",(0,i.kt)("inlineCode",{parentName:"p"},"NERModule.model")," property after initializing weights:\nOption 1: Load weights from specified files mentioned in YAML config\nmodel:\nmodel_config_path\nmodel_config_file\nmodel_path\nmodel_file\nOption 2: Load from Huggingface model hub, specify string in YAML config as:\nmodel:\nhf_model"),(0,i.kt)("h4",{id:"setup"},"setup"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"}," | setup()\n")),(0,i.kt)("p",null,"Method to be called to use plugin out of box. This method will complete preprocessing , create datasets\nsetup the module interface and trainer."))}c.isMDXComponent=!0}}]);