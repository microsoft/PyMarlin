from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# torch installed seperately
required = ['tqdm','tensorboard','azureml-core','pyyaml']
extras = {
    'dev': ['pytest','pylint'],
    'models': ['transformers'],
    'plugins': ['pandas','matplotlib','sklearn','scipy','rouge-score']
}

setup(
    name="pymarlin",
    version="0.3.4",
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