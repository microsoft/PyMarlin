# Documentation

| Owner | Approvers |
| - | - |
| [Amin Saied](mailto:amsaied@microsoft.com) | [Krishan Subudhi](mailto:krkusuk@microsoft.com) |

Documentation is a core aspect of the marlin library.

## Sphinx

We propose using Sphinx to generate our documentation:

- Sphinx is powerful (lots of configuration)
- Sphinx is the _de facto_ standard
- Sphinx fits with CI/CD

Concerns:

- Sphinx uses `.rst` format, which is a bit heavier than md

Mitigation:

- We have added an extension to allow direct support for `.md` files

## Design

The documentation has two components:

1. Auto-generated docstrings: Any changes to the source code are automatically
    picked up when running `sphinx-apidoc` command.
2. Manually created `.md` files: Any additional content - e.g. quickstart, installation
    guides - can be added as markdown files, and declared in the `index.rst`

This design means all the marlin documentation lives in a single location with
flexible layout specified in `index.rst`.