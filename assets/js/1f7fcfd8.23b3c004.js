(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6149],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return k}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},s=Object.keys(e);for(r=0;r<s.length;r++)n=s[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(r=0;r<s.length;r++)n=s[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var i=r.createContext({}),o=function(e){var t=r.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=o(e.components);return r.createElement(i.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,s=e.originalType,i=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),u=o(n),k=a,m=u["".concat(i,".").concat(k)]||u[k]||d[k]||s;return n?r.createElement(m,l(l({ref:t},p),{},{components:n})):r.createElement(m,l({ref:t},p))}));function k(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var s=n.length,l=new Array(s);l[0]=u;var c={};for(var i in t)hasOwnProperty.call(t,i)&&(c[i]=t[i]);c.originalType=e,c.mdxType="string"==typeof e?e:a,l[1]=c;for(var o=2;o<s;o++)l[o]=n[o];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},5121:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return l},metadata:function(){return c},toc:function(){return i},default:function(){return p}});var r=n(2122),a=n(9756),s=(n(7294),n(3905)),l={sidebar_label:"trainer_backend",title:"core.trainer_backend"},c={unversionedId:"reference/core/trainer_backend",id:"reference/core/trainer_backend",isDocsHomePage:!1,title:"core.trainer_backend",description:"Trainer Backend module:",source:"@site/docs/reference/core/trainer_backend.md",sourceDirName:"reference/core",slug:"/reference/core/trainer_backend",permalink:"/PyMarlin/docs/reference/core/trainer_backend",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/core/trainer_backend.md",version:"current",sidebar_label:"trainer_backend",frontMatter:{sidebar_label:"trainer_backend",title:"core.trainer_backend"},sidebar:"referenceSideBar",previous:{title:"core.trainer",permalink:"/PyMarlin/docs/reference/core/trainer"},next:{title:"plugins.hf_ner.data_classes",permalink:"/PyMarlin/docs/reference/plugins/hf_ner/data_classes"}},i=[{value:"TrainerBackendArguments Objects",id:"trainerbackendarguments-objects",children:[]},{value:"TrainerBackend Objects",id:"trainerbackend-objects",children:[]},{value:"OutputCollector Objects",id:"outputcollector-objects",children:[]},{value:"SingleProcess Objects",id:"singleprocess-objects",children:[]},{value:"SingleProcessDpSgd Objects",id:"singleprocessdpsgd-objects",children:[]},{value:"SingleProcessAmp Objects",id:"singleprocessamp-objects",children:[]},{value:"SingleProcessApexAmp Objects",id:"singleprocessapexamp-objects",children:[]},{value:"AbstractTrainerBackendDecorator Objects",id:"abstracttrainerbackenddecorator-objects",children:[]},{value:"DDPTrainerBackend Objects",id:"ddptrainerbackend-objects",children:[]},{value:"DPDDPTrainerBackend Objects",id:"dpddptrainerbackend-objects",children:[]}],o={toc:i};function p(e){var t=e.components,n=(0,a.Z)(e,["components"]);return(0,s.kt)("wrapper",(0,r.Z)({},o,n,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("p",null,"Trainer Backend module:"),(0,s.kt)("p",null,"Currently we support:"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"1. SingleProcess\n2. SingleProcess Amp\n3. SingleProcess Apex-Amp\n4. DDP\n5. DDP Amp\n6. DDP Apex-Amp\n")),(0,s.kt)("p",null,"These are ",(0,s.kt)("inlineCode",{parentName:"p"},"TrainerBackends")," for most common scenarios available out of the box.\nAlternatively a user can provide a custom ",(0,s.kt)("inlineCode",{parentName:"p"},"TrainerBackend"),"."),(0,s.kt)("h4",{id:"build_trainer_backend"},"build","_","trainer","_","backend"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def build_trainer_backend(trainer_backend_name, *args, **kwargs)\n")),(0,s.kt)("p",null,"Factory for trainer_backends"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"trainer_backend_name")," ",(0,s.kt)("em",{parentName:"li"},"str")," - TrainerBackend Name. Possible choices are currently: sp, sp-amp, sp-amp-apex, ddp, ddp-amp, ddp-amp-apex"),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"args")," ",(0,s.kt)("em",{parentName:"li"},"sequence")," - TrainerBackend positional arguments"),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"kwargs")," ",(0,s.kt)("em",{parentName:"li"},"dict")," - TrainerBackend keyword arguments")),(0,s.kt)("h2",{id:"trainerbackendarguments-objects"},"TrainerBackendArguments Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"@dataclasses.dataclass\nclass TrainerBackendArguments()\n")),(0,s.kt)("p",null,"Trainer Backend Arguments dataclass."),(0,s.kt)("h2",{id:"trainerbackend-objects"},"TrainerBackend Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class TrainerBackend(ABC)\n")),(0,s.kt)("p",null,"Trainer Backend abstract class."),(0,s.kt)("h2",{id:"outputcollector-objects"},"OutputCollector Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class OutputCollector()\n")),(0,s.kt)("p",null,"Responsible for collecting step outputs and stores them in memory across each call.\nConcatinates tensors from all steps across first dimension."),(0,s.kt)("h4",{id:"collect"},"collect"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def collect(outputs: Union[torch.Tensor, Iterable[torch.Tensor]])\n")),(0,s.kt)("p",null,"Coalesces train_step and val_step outputs.\nall tensors concatenated across dimension 0\nif input is a torch.Tensor of dimension batch_size ",(0,s.kt)("em",{parentName:"p"}," x")," y .., all_outputs will be List","[torch.Tensor of dimension total_samples_till_now ",(0,s.kt)("em",{parentName:"p"},"x "),"y]","\nif input is a torch.Tensor of dimension 1 ",(0,s.kt)("em",{parentName:"p"}," 1, all_outputs will List[torch.Tensor of dimension total_samples_till_now ")," 1]\nif input is List","[torch.Tensor]",", all_outputs will be List","[torch.Tensor]"," - all tensors concatenated across dimension 0"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"outputs")," ",(0,s.kt)("em",{parentName:"li"},"Union[torch.Tensor, Iterable","[torch.Tensor]","]")," - train_step , val_step outputs")),(0,s.kt)("h2",{id:"singleprocess-objects"},"SingleProcess Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class SingleProcess(TrainerBackend)\n")),(0,s.kt)("p",null,"Single Process Trainer Backend"),(0,s.kt)("h4",{id:"__init__"},"_","_","init","_","_"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def __init__()\n")),(0,s.kt)("p",null,"Single process trainer_backend"),(0,s.kt)("h4",{id:"process_global_step"},"process","_","global","_","step"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def process_global_step(global_step_collector, callback)\n")),(0,s.kt)("p",null,"Clip gradients and call optimizer + scheduler"),(0,s.kt)("h4",{id:"get_state"},"get","_","state"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def get_state() -> dict\n")),(0,s.kt)("p",null,"Get the current state of the trainer_backend, used for checkpointing."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Returns"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"state_dict")," ",(0,s.kt)("em",{parentName:"li"},"dict")," - Dictionary of variables or objects to checkpoint.")),(0,s.kt)("h4",{id:"update_state"},"update","_","state"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def update_state(state) -> None\n")),(0,s.kt)("p",null,"Update the trainer_backend from a checkpointed state."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,s.kt)("p",null,"  state (dict) : Output of get_state() during checkpointing"),(0,s.kt)("h2",{id:"singleprocessdpsgd-objects"},"SingleProcessDpSgd Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class SingleProcessDpSgd(SingleProcess)\n")),(0,s.kt)("p",null,"Backend which supports Differential Privacy. We are using Opacus library.\n",(0,s.kt)("a",{parentName:"p",href:"https://opacus.ai/api/privacy_engine.html"},"https://opacus.ai/api/privacy_engine.html")),(0,s.kt)("h2",{id:"singleprocessamp-objects"},"SingleProcessAmp Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class SingleProcessAmp(SingleProcess)\n")),(0,s.kt)("p",null,"SingleProcess + Native PyTorch AMP Trainer Backend"),(0,s.kt)("h2",{id:"singleprocessapexamp-objects"},"SingleProcessApexAmp Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class SingleProcessApexAmp(SingleProcessAmp)\n")),(0,s.kt)("p",null,"SingleProcess + Apex AMP Trainer Backend"),(0,s.kt)("h2",{id:"abstracttrainerbackenddecorator-objects"},"AbstractTrainerBackendDecorator Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class AbstractTrainerBackendDecorator(TrainerBackend)\n")),(0,s.kt)("p",null,"Abstract class implementing the decorator design pattern."),(0,s.kt)("h2",{id:"ddptrainerbackend-objects"},"DDPTrainerBackend Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class DDPTrainerBackend(AbstractTrainerBackendDecorator)\n")),(0,s.kt)("p",null,"Distributed Data Parallel TrainerBackend."),(0,s.kt)("p",null,"Wraps ModuleInterface model with DistributedDataParallel which handles\ngradient averaging across processes."),(0,s.kt)("p",null,".. note: Assumes initiailized model parameters are consistent across\nprocesses - e.g. by using same random seed in each process at\npoint of model initialization."),(0,s.kt)("h4",{id:"setup_distributed_env"},"setup","_","distributed","_","env"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def setup_distributed_env()\n")),(0,s.kt)("p",null,"Setup the process group for distributed training."),(0,s.kt)("h4",{id:"cleanup"},"cleanup"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def cleanup()\n")),(0,s.kt)("p",null,"Destroy the process group used for distributed training."),(0,s.kt)("h4",{id:"gather_tensors_on_cpu"},"gather","_","tensors","_","on","_","cpu"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def gather_tensors_on_cpu(x: torch.tensor)\n")),(0,s.kt)("p",null,"Gather tensors and move to cpu at configurable frequency."),(0,s.kt)("p",null,"Move tensor to CUDA device, apply all-gather and move back to CPU.\nIf ",(0,s.kt)("inlineCode",{parentName:"p"},"distributed_training_args.gather_frequency")," is set,  tensors are\nmoved to CUDA in chunks of that size."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("inlineCode",{parentName:"li"},"x")," ",(0,s.kt)("em",{parentName:"li"},"torch.tensor")," - To be gathered.")),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Returns"),":"),(0,s.kt)("p",null,"  Gathered tensor on the cpu."),(0,s.kt)("h2",{id:"dpddptrainerbackend-objects"},"DPDDPTrainerBackend Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class DPDDPTrainerBackend(DDPTrainerBackend)\n")),(0,s.kt)("p",null,"Distributed Data Parallel TrainerBackend with Differential Privacy."),(0,s.kt)("p",null,"Wraps ModuleInterface model with DifferentiallyPrivateDistributedDataParallel which handles\ngradient averaging across processes, along with virtual stepping."),(0,s.kt)("p",null,".. note: Assumes initiailized model parameters are consistent across\nprocesses - e.g. by using same random seed in each process at\npoint of model initialization."))}p.isMDXComponent=!0}}]);