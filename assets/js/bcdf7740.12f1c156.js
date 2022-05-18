(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7580],{3905:function(e,r,t){"use strict";t.d(r,{Zo:function(){return s},kt:function(){return f}});var n=t(7294);function l(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function i(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function a(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?i(Object(t),!0).forEach((function(r){l(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function o(e,r){if(null==e)return{};var t,n,l=function(e,r){if(null==e)return{};var t,n,l={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(l[t]=e[t]);return l}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(l[t]=e[t])}return l}var c=n.createContext({}),u=function(e){var r=n.useContext(c),t=r;return e&&(t="function"==typeof e?e(r):a(a({},r),e)),t},s=function(e){var r=u(e.components);return n.createElement(c.Provider,{value:r},e.children)},p={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},m=n.forwardRef((function(e,r){var t=e.components,l=e.mdxType,i=e.originalType,c=e.parentName,s=o(e,["components","mdxType","originalType","parentName"]),m=u(t),f=l,d=m["".concat(c,".").concat(f)]||m[f]||p[f]||i;return t?n.createElement(d,a(a({ref:r},s),{},{components:t})):n.createElement(d,a({ref:r},s))}));function f(e,r){var t=arguments,l=r&&r.mdxType;if("string"==typeof e||l){var i=t.length,a=new Array(i);a[0]=m;var o={};for(var c in r)hasOwnProperty.call(r,c)&&(o[c]=r[c]);o.originalType=e,o.mdxType="string"==typeof e?e:l,a[1]=o;for(var u=2;u<i;u++)a[u]=t[u];return n.createElement.apply(null,a)}return n.createElement.apply(null,t)}m.displayName="MDXCreateElement"},2837:function(e,r,t){"use strict";t.r(r),t.d(r,{frontMatter:function(){return a},metadata:function(){return o},toc:function(){return c},default:function(){return s}});var n=t(2122),l=t(9756),i=(t(7294),t(3905)),a={sidebar_label:"aml",title:"utils.writer.aml"},o={unversionedId:"reference/utils/writer/aml",id:"reference/utils/writer/aml",isDocsHomePage:!1,title:"utils.writer.aml",description:"AML writer module.",source:"@site/docs/reference/utils/writer/aml.md",sourceDirName:"reference/utils/writer",slug:"/reference/utils/writer/aml",permalink:"/PyMarlin/docs/reference/utils/writer/aml",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/utils/writer/aml.md",version:"current",sidebar_label:"aml",frontMatter:{sidebar_label:"aml",title:"utils.writer.aml"},sidebar:"referenceSideBar",previous:{title:"utils.writer",permalink:"/PyMarlin/docs/reference/utils/writer/__init__"},next:{title:"utils.writer.base",permalink:"/PyMarlin/docs/reference/utils/writer/base"}},c=[{value:"Aml Objects",id:"aml-objects",children:[]}],u={toc:c};function s(e){var r=e.components,t=(0,l.Z)(e,["components"]);return(0,i.kt)("wrapper",(0,n.Z)({},u,t,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("p",null,"AML writer module."),(0,i.kt)("h2",{id:"aml-objects"},"Aml Objects"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"class Aml(Writer)\n")),(0,i.kt)("p",null,"This class implements the Azure ML writer for stats."),(0,i.kt)("h4",{id:"log_scalar"},"log","_","scalar"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"def log_scalar(k, v, step)\n")),(0,i.kt)("p",null,"Log metric to AML."),(0,i.kt)("h4",{id:"log_multi"},"log","_","multi"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"def log_multi(k, v, step)\n")),(0,i.kt)("p",null,"Log metrics to stdout."))}s.isMDXComponent=!0}}]);