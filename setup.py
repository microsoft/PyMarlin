from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# torch installed seperately
required = ['tqdm','tensorboard','azureml-core','pyyaml','pandas']
extras = {
    'dev': ['pytest','pylint'],
}

setup(
    name="pymarlin",
    version="0.3.2",
    author="ELR Team",
    author_email="elrcore@microsoft.com",
    description="Lightweight Deeplearning Library",
    long_description=long_description,
    url="https://aka.ms/marlin/docs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    extras_require=extras
)

# https://packaging.python.org/discussions/install-requires-vs-requirements/