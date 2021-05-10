# Set up instructions

## Environment setup
    conda create -n pymarlin python=3.8
    conda activate pymarlin
    conda install pytorch cpuonly -c pytorch

# Installation

## Install from pip package

    pip install pymarlin

### Test
    python -c 'import pymarlin as ml; help(ml)'
    Hello World

## Install from source
    git clone https://github.com/microsoft/PyMarlin.git
    cd PyMarlin

### Option 1: pip install 

    pip install .
    cd .. 
    python 

    Hello World

### Option 2: PYTHONPATH
    set PYTHONPATH=<sourcecode path>

## Developing marlin
1. Run pylint
    https://docs.pylint.org/en/1.8/user_guide/run.html

    Get exit code in windows: https://www.shellhacks.com/windows-get-exit-code-errorlevel-cmd-powershell/

        > pylint --rcfile .pylintrc marlin

        > $LastExitCode #make sure it's 0


    Enable linting on VScode : https://code.visualstudio.com/docs/python/linting

    Tip: conda environment must be selected and . `.pylint` rc file should be at the root of workspace

2. Run test cases
        
        pip install pytest
        python -m pytest tests

## Publish and install pip package

Document reference:

Official documentation:https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops
Our feed where packages will be stored: https://o365exchange.visualstudio.com/O365%20Core/_packaging?_a=connect&feed=marlinpi

https://github.com/microsoft/artifacts-keyring

### Publish

https://packaging.python.org/tutorials/packaging-projects/


    python -m pip install --upgrade build
    python -m build

This command should output a lot of text and once completed should generate two files in the dist directory

    python -m pip install --user --upgrade twine
    python -m twine upload --repository testpypi dist/* --skip-existing

### install
    python -m pip install --index-url https://test.pypi.org/simple/ --no-deps pymarlin