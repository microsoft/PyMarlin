(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5208],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return u},kt:function(){return d}});var r=t(7294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function s(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function i(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?s(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):s(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},s=Object.keys(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(r=0;r<s.length;r++)t=s[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var o=r.createContext({}),c=function(e){var n=r.useContext(o),t=n;return e&&(t="function"==typeof e?e(n):i(i({},n),e)),t},u=function(e){var n=c(e.components);return r.createElement(o.Provider,{value:n},e.children)},p={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},f=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,s=e.originalType,o=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),f=c(t),d=a,m=f["".concat(o,".").concat(d)]||f[d]||p[d]||s;return t?r.createElement(m,i(i({ref:n},u),{},{components:t})):r.createElement(m,i({ref:n},u))}));function d(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var s=t.length,i=new Array(s);i[0]=f;var l={};for(var o in n)hasOwnProperty.call(n,o)&&(l[o]=n[o]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var c=2;c<s;c++)i[c]=t[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,t)}f.displayName="MDXCreateElement"},3162:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return i},metadata:function(){return l},toc:function(){return o},default:function(){return u}});var r=t(2122),a=t(9756),s=(t(7294),t(3905)),i={sidebar_label:"data_classes",title:"plugins.ner_plugin.data_classes"},l={unversionedId:"reference/plugins/ner_plugin/data_classes",id:"reference/plugins/ner_plugin/data_classes",isDocsHomePage:!1,title:"plugins.ner_plugin.data_classes",description:"NERProcessor Objects",source:"@site/docs/reference/plugins/ner_plugin/data_classes.md",sourceDirName:"reference/plugins/ner_plugin",slug:"/reference/plugins/ner_plugin/data_classes",permalink:"/docs/reference/plugins/ner_plugin/data_classes",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/plugins/ner_plugin/data_classes.md",version:"current",sidebar_label:"data_classes",frontMatter:{sidebar_label:"data_classes",title:"plugins.ner_plugin.data_classes"},sidebar:"referenceSideBar",previous:{title:"plugins.hf_seq_classification.module_classes",permalink:"/docs/reference/plugins/hf_seq_classification/module_classes"},next:{title:"plugins.ner_plugin.implementation",permalink:"/docs/reference/plugins/ner_plugin/implementation"}},o=[{value:"NERProcessor Objects",id:"nerprocessor-objects",children:[]}],c={toc:o};function u(e){var n=e.components,t=(0,a.Z)(e,["components"]);return(0,s.kt)("wrapper",(0,r.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,s.kt)("h2",{id:"nerprocessor-objects"},"NERProcessor Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class NERProcessor(data_interface.DataProcessor)\n")),(0,s.kt)("p",null,"Reads a file (tsv/csv) line by line and tokenizes text using Huggingface AutoTokenizer.\nRequires header ",'"',"Sentence",'"'," and ",'"',"Slot",'"'," for the text and corresponding labels"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"args")," ",(0,s.kt)("em",{parentName:"li"},"arguments.DataArguments")," - Dataclass")),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Returns"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"features")," ",(0,s.kt)("em",{parentName:"li"},"List","[transformers.InputFeatures]")," - List of tokenized features.")))}u.isMDXComponent=!0}}]);