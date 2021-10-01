from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

required = ['torch<=1.9.1','tqdm','tensorboard', 'Pillow','azureml-core','pyyaml','pandas']
extras = {
    'dev': ['pylint', 'pytest', 'pytest-cov'],
    'plugins': ['transformers','pandas','matplotlib','sklearn','scipy','rouge-score']
}

setup(
    name="pymarlin",
    version="0.2.8",
    author="ELR Team",
    author_email="elrcore@microsoft.com",
    description="Lightweight Deeplearning Library",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://microsoft.github.io/PyMarlin/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    extras_require=extras,
    python_requires=">=3.6",
)
# https://packaging.python.org/discussions/install-requires-vs-requirements/