(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8882],{3905:function(e,r,t){"use strict";t.d(r,{Zo:function(){return s},kt:function(){return m}});var n=t(7294);function i(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function o(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function a(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?o(Object(t),!0).forEach((function(r){i(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function c(e,r){if(null==e)return{};var t,n,i=function(e,r){if(null==e)return{};var t,n,i={},o=Object.keys(e);for(n=0;n<o.length;n++)t=o[n],r.indexOf(t)>=0||(i[t]=e[t]);return i}(e,r);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)t=o[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var l=n.createContext({}),u=function(e){var r=n.useContext(l),t=r;return e&&(t="function"==typeof e?e(r):a(a({},r),e)),t},s=function(e){var r=u(e.components);return n.createElement(l.Provider,{value:r},e.children)},p={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},f=n.forwardRef((function(e,r){var t=e.components,i=e.mdxType,o=e.originalType,l=e.parentName,s=c(e,["components","mdxType","originalType","parentName"]),f=u(t),m=i,d=f["".concat(l,".").concat(m)]||f[m]||p[m]||o;return t?n.createElement(d,a(a({ref:r},s),{},{components:t})):n.createElement(d,a({ref:r},s))}));function m(e,r){var t=arguments,i=r&&r.mdxType;if("string"==typeof e||i){var o=t.length,a=new Array(o);a[0]=f;var c={};for(var l in r)hasOwnProperty.call(r,l)&&(c[l]=r[l]);c.originalType=e,c.mdxType="string"==typeof e?e:i,a[1]=c;for(var u=2;u<o;u++)a[u]=t[u];return n.createElement.apply(null,a)}return n.createElement.apply(null,t)}f.displayName="MDXCreateElement"},1042:function(e,r,t){"use strict";t.r(r),t.d(r,{frontMatter:function(){return a},metadata:function(){return c},toc:function(){return l},default:function(){return s}});var n=t(2122),i=t(9756),o=(t(7294),t(3905)),a={sidebar_label:"writer",title:"utils.writer"},c={unversionedId:"reference/utils/writer/__init__",id:"reference/utils/writer/__init__",isDocsHomePage:!1,title:"utils.writer",description:"Writers package.",source:"@site/docs/reference/utils/writer/__init__.md",sourceDirName:"reference/utils/writer",slug:"/reference/utils/writer/__init__",permalink:"/docs/reference/utils/writer/__init__",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/utils/writer/__init__.md",version:"current",sidebar_label:"writer",frontMatter:{sidebar_label:"writer",title:"utils.writer"},sidebar:"referenceSideBar",previous:{title:"utils.stats.basic_stats",permalink:"/docs/reference/utils/stats/basic_stats"},next:{title:"utils.writer.aml",permalink:"/docs/reference/utils/writer/aml"}},l=[],u={toc:l};function s(e){var r=e.components,t=(0,i.Z)(e,["components"]);return(0,o.kt)("wrapper",(0,n.Z)({},u,t,{components:r,mdxType:"MDXLayout"}),(0,o.kt)("p",null,"Writers package."),(0,o.kt)("h4",{id:"build_writer"},"build","_","writer"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"build_writer(writer, args: WriterInitArguments)\n")),(0,o.kt)("p",null,"Initializes and returns writer object based on writer type."))}s.isMDXComponent=!0}}]);