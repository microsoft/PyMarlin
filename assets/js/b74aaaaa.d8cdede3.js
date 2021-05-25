(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6320],{3905:function(e,t,r){"use strict";r.d(t,{Zo:function(){return c},kt:function(){return m}});var n=r(7294);function l(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){l(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function a(e,t){if(null==e)return{};var r,n,l=function(e,t){if(null==e)return{};var r,n,l={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(l[r]=e[r]);return l}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(l[r]=e[r])}return l}var s=n.createContext({}),u=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},c=function(e){var t=u(e.components);return n.createElement(s.Provider,{value:t},e.children)},g={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},p=n.forwardRef((function(e,t){var r=e.components,l=e.mdxType,i=e.originalType,s=e.parentName,c=a(e,["components","mdxType","originalType","parentName"]),p=u(r),m=l,f=p["".concat(s,".").concat(m)]||p[m]||g[m]||i;return r?n.createElement(f,o(o({ref:t},c),{},{components:r})):n.createElement(f,o({ref:t},c))}));function m(e,t){var r=arguments,l=t&&t.mdxType;if("string"==typeof e||l){var i=r.length,o=new Array(i);o[0]=p;var a={};for(var s in t)hasOwnProperty.call(t,s)&&(a[s]=t[s]);a.originalType=e,a.mdxType="string"==typeof e?e:l,o[1]=a;for(var u=2;u<i;u++)o[u]=r[u];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}p.displayName="MDXCreateElement"},144:function(e,t,r){"use strict";r.r(t),r.d(t,{frontMatter:function(){return o},metadata:function(){return a},toc:function(){return s},default:function(){return c}});var n=r(2122),l=r(9756),i=(r(7294),r(3905)),o={sidebar_label:"logging_utils",title:"utils.logger.logging_utils"},a={unversionedId:"reference/utils/logger/logging_utils",id:"reference/utils/logger/logging_utils",isDocsHomePage:!1,title:"utils.logger.logging_utils",description:"Logging util module",source:"@site/docs/reference/utils/logger/logging_utils.md",sourceDirName:"reference/utils/logger",slug:"/reference/utils/logger/logging_utils",permalink:"/docs/reference/utils/logger/logging_utils",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/utils/logger/logging_utils.md",version:"current",sidebar_label:"logging_utils",frontMatter:{sidebar_label:"logging_utils",title:"utils.logger.logging_utils"},sidebar:"referenceSideBar",previous:{title:"utils.config_parser.custom_arg_parser",permalink:"/docs/reference/utils/config_parser/custom_arg_parser"},next:{title:"utils.misc.misc_utils",permalink:"/docs/reference/utils/misc/misc_utils"}},s=[],u={toc:s};function c(e){var t=e.components,r=(0,l.Z)(e,["components"]);return(0,i.kt)("wrapper",(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,"Logging util module"),(0,i.kt)("h4",{id:"getlogger"},"getlogger"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"getlogger(name, log_level='INFO')\n")),(0,i.kt)("p",null,"This method returns a logger object to be used by the calling class.\nThe logger object returned has the following format for all the logs:\n","'","SystemLog: %(asctime)s:%(levelname)s : %(name)s : %(lineno)d : %(message)s","'"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"name")," ",(0,i.kt)("em",{parentName:"li"},"str")," - Directory under which to search for checkpointed files."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"file_prefix")," ",(0,i.kt)("em",{parentName:"li"},"str")," - Prefix to match for when searching for candidate files."),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"file_ext")," ",(0,i.kt)("em",{parentName:"li"},"str, optional")," - File extension to consider when searching.")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Returns"),":"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"logger")," ",(0,i.kt)("em",{parentName:"li"},"object")," - logger object to use for logging.")))}c.isMDXComponent=!0}}]);