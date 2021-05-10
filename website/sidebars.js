/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {
  docsSidebar: {
    'Getting Started': ['getting-started', 'installation', 'marlin-in-pictures'],
    'Examples': ['examples/bart', 'examples/cifar', 'examples/datamodule-example', 'examples/distillation', 'examples/glue-tasks', 'examples/ner'],
    'Utils': ['utils/stats'],
    'Feedback & Credits': ['credits']
  },
  // pydoc-markdown auto-generated markdowns from docstrings
  referenceSideBar: require.resolve("./docs/reference/sidebar.json")
};
