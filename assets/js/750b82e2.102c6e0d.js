(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4288],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return u},kt:function(){return m}});var r=t(7294);function l(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){l(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,r,l=function(e,n){if(null==e)return{};var t,r,l={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(l[t]=e[t]);return l}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(l[t]=e[t])}return l}var s=r.createContext({}),c=function(e){var n=r.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},u=function(e){var n=c(e.components);return r.createElement(s.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},d=r.forwardRef((function(e,n){var t=e.components,l=e.mdxType,a=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),d=c(t),m=l,f=d["".concat(s,".").concat(m)]||d[m]||p[m]||a;return t?r.createElement(f,o(o({ref:n},u),{},{components:t})):r.createElement(f,o({ref:n},u))}));function m(e,n){var t=arguments,l=n&&n.mdxType;if("string"==typeof e||l){var a=t.length,o=new Array(a);o[0]=d;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i.mdxType="string"==typeof e?e:l,o[1]=i;for(var c=2;c<a;c++)o[c]=t[c];return r.createElement.apply(null,o)}return r.createElement.apply(null,t)}d.displayName="MDXCreateElement"},9439:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return o},metadata:function(){return i},toc:function(){return s},default:function(){return u}});var r=t(2122),l=t(9756),a=(t(7294),t(3905)),o={sidebar_label:"module_classes",title:"plugins.hf_ner.module_classes"},i={unversionedId:"reference/plugins/hf_ner/module_classes",id:"reference/plugins/hf_ner/module_classes",isDocsHomePage:!1,title:"plugins.hf_ner.module_classes",description:"NERModule Objects",source:"@site/docs/reference/plugins/hf_ner/module_classes.md",sourceDirName:"reference/plugins/hf_ner",slug:"/reference/plugins/hf_ner/module_classes",permalink:"/docs/reference/plugins/hf_ner/module_classes",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/plugins/hf_ner/module_classes.md",version:"current",sidebar_label:"module_classes",frontMatter:{sidebar_label:"module_classes",title:"plugins.hf_ner.module_classes"},sidebar:"referenceSideBar",previous:{title:"plugins.hf_ner.implementation",permalink:"/docs/reference/plugins/hf_ner/implementation"},next:{title:"plugins.hf_ner.sequence_labelling_metrics",permalink:"/docs/reference/plugins/hf_ner/sequence_labelling_metrics"}},s=[{value:"NERModule Objects",id:"nermodule-objects",children:[]}],c={toc:s};function u(e){var n=e.components,t=(0,l.Z)(e,["components"]);return(0,a.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h2",{id:"nermodule-objects"},"NERModule Objects"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"class NERModule(module_interface.ModuleInterface)\n")),(0,a.kt)("p",null,"NER Task specific ModuleInterface used with a trainer.\nThe ",(0,a.kt)("inlineCode",{parentName:"p"},"data")," and ",(0,a.kt)("inlineCode",{parentName:"p"},"model")," are required properties and must be set."),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,a.kt)("p",null,"  ModuleInterfaceArguments : contains module interface arguments , i.e. max learning rate,\nwarmup propotion, type of trainer , etc. Also includes modelArguments class as attribute\nwhich include model specific arguments such as hfmodel name , modep path , model file name , etc"),(0,a.kt)("h4",{id:"data"},"data"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"}," | @property\n | data()\n")),(0,a.kt)("p",null,"DataInterface object that is used to retrieve corresponding train or val dataset."),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Returns"),":"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"data")," - DataInterface object with at least one of train or val data.")),(0,a.kt)("h4",{id:"model"},"model"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"}," | @property\n | model()\n")),(0,a.kt)("p",null,"Pytorch model."),(0,a.kt)("h4",{id:"setup_model"},"setup","_","model"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"}," | setup_model(model_class)\n")),(0,a.kt)("p",null,"Initializes ",(0,a.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModule.model")," weights:\nOption 1: Load weights from specified files mentioned in YAML config\nmodel:\nmodel_config_path\nmodel_config_file\nmodel_path\nmodel_file\nOption 2: Load from Huggingface model hub, specify string in YAML config as:\nmodel:\nhf_model\nIf distill_args.enable = True\nstudent = ",(0,a.kt)("inlineCode",{parentName:"p"},"NERModule.model"),"\nteacher = ",(0,a.kt)("inlineCode",{parentName:"p"},"NERModule.teacher")),(0,a.kt)("p",null,(0,a.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("inlineCode",{parentName:"li"},"automodel_class")," - Huggingface AutoModelForTokenClassificaton class")))}u.isMDXComponent=!0}}]);