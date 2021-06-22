(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4280],{3905:function(e,t,r){"use strict";r.d(t,{Zo:function(){return u},kt:function(){return k}});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function l(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function s(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?l(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):l(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},l=Object.keys(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var p=n.createContext({}),o=function(e){var t=n.useContext(p),r=t;return e&&(r="function"==typeof e?e(t):s(s({},t),e)),r},u=function(e){var t=o(e.components);return n.createElement(p.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,l=e.originalType,p=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),m=o(r),k=a,d=m["".concat(p,".").concat(k)]||m[k]||c[k]||l;return r?n.createElement(d,s(s({ref:t},u),{},{components:r})):n.createElement(d,s({ref:t},u))}));function k(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=r.length,s=new Array(l);s[0]=m;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i.mdxType="string"==typeof e?e:a,s[1]=i;for(var o=2;o<l;o++)s[o]=r[o];return n.createElement.apply(null,s)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},7e3:function(e,t,r){"use strict";r.r(t),r.d(t,{frontMatter:function(){return s},metadata:function(){return i},toc:function(){return p},default:function(){return u}});var n=r(2122),a=r(9756),l=(r(7294),r(3905)),s={sidebar_label:"sequence_labelling_metrics",title:"plugins.hf_ner.sequence_labelling_metrics"},i={unversionedId:"reference/plugins/hf_ner/sequence_labelling_metrics",id:"reference/plugins/hf_ner/sequence_labelling_metrics",isDocsHomePage:!1,title:"plugins.hf_ner.sequence_labelling_metrics",description:"Metrics to assess performance on sequence labeling task given prediction",source:"@site/docs/reference/plugins/hf_ner/sequence_labelling_metrics.md",sourceDirName:"reference/plugins/hf_ner",slug:"/reference/plugins/hf_ner/sequence_labelling_metrics",permalink:"/PyMarlin/docs/reference/plugins/hf_ner/sequence_labelling_metrics",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/plugins/hf_ner/sequence_labelling_metrics.md",version:"current",sidebar_label:"sequence_labelling_metrics",frontMatter:{sidebar_label:"sequence_labelling_metrics",title:"plugins.hf_ner.sequence_labelling_metrics"},sidebar:"referenceSideBar",previous:{title:"plugins.hf_ner.module_classes",permalink:"/PyMarlin/docs/reference/plugins/hf_ner/module_classes"},next:{title:"plugins.hf_seq2seq.data_classes",permalink:"/PyMarlin/docs/reference/plugins/hf_seq2seq/data_classes"}},p=[],o={toc:p};function u(e){var t=e.components,r=(0,a.Z)(e,["components"]);return(0,l.kt)("wrapper",(0,n.Z)({},o,r,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("p",null,"Metrics to assess performance on sequence labeling task given prediction\nFunctions named as ",(0,l.kt)("inlineCode",{parentName:"p"},"*_score")," return a scalar value to maximize: the higher\nthe better"),(0,l.kt)("h4",{id:"get_entities"},"get","_","entities"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"get_entities(seq, suffix=False)\n")),(0,l.kt)("p",null,"Gets entities from sequence."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"seq")," ",(0,l.kt)("em",{parentName:"li"},"list")," - sequence of labels.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"list")," - list of (chunk_type, chunk_start, chunk_end).")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics.sequence_labeling import get_entities\nseq = ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","'",", ","'","B-LOC","']","\nget_entities(seq)\n","[(","'","PER","'",", 0, 1), (","'","LOC","'",", 3, 3)]"),(0,l.kt)("h4",{id:"end_of_chunk"},"end","_","of","_","chunk"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"end_of_chunk(prev_tag, tag, prev_type, type_)\n")),(0,l.kt)("p",null,"Checks if a chunk ended between the previous and current word."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"prev_tag")," - previous chunk tag."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"tag")," - current chunk tag."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"prev_type")," - previous type."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"type_")," - current type.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"chunk_end")," - boolean.")),(0,l.kt)("h4",{id:"start_of_chunk"},"start","_","of","_","chunk"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"start_of_chunk(prev_tag, tag, prev_type, type_)\n")),(0,l.kt)("p",null,"Checks if a chunk started between the previous and current word."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"prev_tag")," - previous chunk tag."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"tag")," - current chunk tag."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"prev_type")," - previous type."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"type_")," - current type.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"chunk_start")," - boolean.")),(0,l.kt)("h4",{id:"f1_score"},"f1","_","score"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'f1_score(y_true, y_pred, average="micro", suffix=False)\n')),(0,l.kt)("p",null,"Compute the F1 score.\nThe F1 score can be interpreted as a weighted average of the precision and\nrecall, where an F1 score reaches its best value at 1 and worst score at 0.\nThe relative contribution of precision and recall to the F1 score are\nequal. The formula for the F1 score is::\nF1 = 2 ",(0,l.kt)("em",{parentName:"p"}," (precision ")," recall) / (precision + recall)"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a tagger."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  score : float."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import f1_score\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\nf1_score(y_true, y_pred)\n0.50"),(0,l.kt)("h4",{id:"accuracy_score"},"accuracy","_","score"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"accuracy_score(y_true, y_pred)\n")),(0,l.kt)("p",null,"Accuracy classification score.\nIn multilabel classification, this function computes subset accuracy:\nthe set of labels predicted for a sample must ",(0,l.kt)("em",{parentName:"p"},"exactly")," match the\ncorresponding set of labels in y_true."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a tagger."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  score : float."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import accuracy_score\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\naccuracy_score(y_true, y_pred)\n0.80"),(0,l.kt)("h4",{id:"precision_score"},"precision","_","score"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'precision_score(y_true, y_pred, average="micro", suffix=False)\n')),(0,l.kt)("p",null,"Compute the precision.\nThe precision is the ratio ",(0,l.kt)("inlineCode",{parentName:"p"},"tp / (tp + fp)")," where ",(0,l.kt)("inlineCode",{parentName:"p"},"tp")," is the number of\ntrue positives and ",(0,l.kt)("inlineCode",{parentName:"p"},"fp")," the number of false positives. The precision is\nintuitively the ability of the classifier not to label as positive a sample.\nThe best value is 1 and the worst value is 0."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a tagger."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  score : float."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import precision_score\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\nprecision_score(y_true, y_pred)\n0.50"),(0,l.kt)("h4",{id:"recall_score"},"recall","_","score"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'recall_score(y_true, y_pred, average="micro", suffix=False)\n')),(0,l.kt)("p",null,"Compute the recall.\nThe recall is the ratio ",(0,l.kt)("inlineCode",{parentName:"p"},"tp / (tp + fn)")," where ",(0,l.kt)("inlineCode",{parentName:"p"},"tp")," is the number of\ntrue positives and ",(0,l.kt)("inlineCode",{parentName:"p"},"fn")," the number of false negatives. The recall is\nintuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a tagger."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  score : float."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import recall_score\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\nrecall_score(y_true, y_pred)\n0.50"),(0,l.kt)("h4",{id:"performance_measure"},"performance","_","measure"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"performance_measure(y_true, y_pred)\n")),(0,l.kt)("p",null,"Compute the performance metrics: TP, FP, FN, TN"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a tagger."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  performance_dict : dict"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Example"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import performance_measure\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","O","'",", ","'","B-ORG","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\nperformance_measure(y_true, y_pred)\n(3, 3, 1, 4)"),(0,l.kt)("h4",{id:"classification_report"},"classification","_","report"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"classification_report(y_true, y_pred, digits=2, suffix=False)\n")),(0,l.kt)("p",null,"Build a text report showing the main classification metrics."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("p",null,"  y_true : 2d array. Ground truth (correct) target values.\ny_pred : 2d array. Estimated targets as returned by a classifier.\ndigits : int. Number of digits for formatting output floating point values."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("p",null,"  report : string. Text summary of the precision, recall, F1 score for each class."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Examples"),":"),(0,l.kt)("p",null,"  from seqeval.metrics import classification_report\ny_true = [","['","O","'",", ","'","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\ny_pred = [","['","O","'",", ","'","O","'",", ","'","B-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","I-MISC","'",", ","'","O","']",", ","['","B-PER","'",", ","'","I-PER","'",", ","'","O","']","]\nprint(classification_report(y_true, y_pred))\nprecision    recall  f1-score   support\n","<","BLANKLINE",">","\nMISC       0.00      0.00      0.00         1\nPER       1.00      1.00      1.00         1\n","<","BLANKLINE",">","\nmicro avg       0.50      0.50      0.50         2\nmacro avg       0.50      0.50      0.50         2\n","<","BLANKLINE",">"))}u.isMDXComponent=!0}}]);