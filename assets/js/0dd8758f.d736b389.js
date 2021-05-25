(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5453],{3905:function(e,r,t){"use strict";t.d(r,{Zo:function(){return u},kt:function(){return b}});var n=t(7294);function i(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function a(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function s(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?a(Object(t),!0).forEach((function(r){i(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function c(e,r){if(null==e)return{};var t,n,i=function(e,r){if(null==e)return{};var t,n,i={},a=Object.keys(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||(i[t]=e[t]);return i}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)t=a[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var o=n.createContext({}),l=function(e){var r=n.useContext(o),t=r;return e&&(t="function"==typeof e?e(r):s(s({},r),e)),t},u=function(e){var r=l(e.components);return n.createElement(o.Provider,{value:r},e.children)},p={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},f=n.forwardRef((function(e,r){var t=e.components,i=e.mdxType,a=e.originalType,o=e.parentName,u=c(e,["components","mdxType","originalType","parentName"]),f=l(t),b=i,m=f["".concat(o,".").concat(b)]||f[b]||p[b]||a;return t?n.createElement(m,s(s({ref:r},u),{},{components:t})):n.createElement(m,s({ref:r},u))}));function b(e,r){var t=arguments,i=r&&r.mdxType;if("string"==typeof e||i){var a=t.length,s=new Array(a);s[0]=f;var c={};for(var o in r)hasOwnProperty.call(r,o)&&(c[o]=r[o]);c.originalType=e,c.mdxType="string"==typeof e?e:i,s[1]=c;for(var l=2;l<a;l++)s[l]=t[l];return n.createElement.apply(null,s)}return n.createElement.apply(null,t)}f.displayName="MDXCreateElement"},9759:function(e,r,t){"use strict";t.r(r),t.d(r,{frontMatter:function(){return s},metadata:function(){return c},toc:function(){return o},default:function(){return u}});var n=t(2122),i=t(9756),a=(t(7294),t(3905)),s={sidebar_label:"base",title:"utils.writer.base"},c={unversionedId:"reference/utils/writer/base",id:"reference/utils/writer/base",isDocsHomePage:!1,title:"utils.writer.base",description:"Base class for Writers",source:"@site/docs/reference/utils/writer/base.md",sourceDirName:"reference/utils/writer",slug:"/reference/utils/writer/base",permalink:"/docs/reference/utils/writer/base",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/utils/writer/base.md",version:"current",sidebar_label:"base",frontMatter:{sidebar_label:"base",title:"utils.writer.base"},sidebar:"referenceSideBar",previous:{title:"utils.writer.aml",permalink:"/docs/reference/utils/writer/aml"},next:{title:"utils.writer.stdout",permalink:"/docs/reference/utils/writer/stdout"}},o=[{value:"WriterInitArguments Objects",id:"writerinitarguments-objects",children:[]},{value:"Writer Objects",id:"writer-objects",children:[]}],l={toc:o};function u(e){var r=e.components,t=(0,i.Z)(e,["components"]);return(0,a.kt)("wrapper",(0,n.Z)({},l,t,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("p",null,"Base class for Writers"),(0,a.kt)("h2",{id:"writerinitarguments-objects"},"WriterInitArguments Objects"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"@dataclasses.dataclass\nclass WriterInitArguments()\n")),(0,a.kt)("p",null,"Writer Arguments."),(0,a.kt)("h2",{id:"writer-objects"},"Writer Objects"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-python"},"class Writer(ABC)\n")),(0,a.kt)("p",null,"Abstract Base class for Writers."))}u.isMDXComponent=!0}}]);