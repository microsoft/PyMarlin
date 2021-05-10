# Set up instructions

## Environment setup
    conda create -n pymarlin python=3.8
    conda activate pymarlin
    conda install pytorch cpuonly -c pytorch

# Installation

## Install from pip package

pip install keyring artifacts-keyring #https://github.com/microsoft/artifacts-keyring
pip install marlin --index-url https://o365exchange.pkgs.visualstudio.com/959adb23-f323-4d52-8203-ff34e5cbeefa/_packaging/marlinpi/pypi/simple --force-reinstall

## Install from source
    git clone https://github.com/microsoft/PyMarlin.git
    cd PyMarlin

### Option 1: pip install 

    pip install .
    cd .. 
    python -c 'import pymarlin as ml; ml.hello(); print(ml.__path__)'
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

    
    Linux based exit handler
    https://github.com/jongracecox/pylint-exit

2. Run test cases
    
        python -m pytest test

## Publish and install pip package

Document reference:

Official documentation:https://docs.microsoft.com/en-us/azure/devops/artifacts/quickstarts/python-packages?view=azure-devops
Our feed where packages will be stored: https://o365exchange.visualstudio.com/O365%20Core/_packaging?_a=connect&feed=marlinpi

https://github.com/microsoft/artifacts-keyring

### Publish

pip install keyring artifacts-keyring
pip install twine



add .pypirc to home directory and write this in it.
    [distutils]
    Index-servers =
    marlinpi

    [marlinpi]
    Repository = https://o365exchange.pkgs.visualstudio.com/959adb23-f323-4d52-8203-ff34e5cbeefa/_packaging/marlinpi/pypi/upload

Go to marlin directory
    cd C:\Users\krkusuk\repos\ELR\sources\dev\SubstrateInferences\marlin\

Run these commands to upload to the feed 
    
    python setup.py sdist bdist_wheel
    twine upload -r marlinpi dist/* # --skip-existing

### install
    conda create -n test2 python=3.8
    pip install keyring artifacts-keyring #https://github.com/microsoft/artifacts-keyring
    pip install marlin --index-url https://o365exchange.pkgs.visualstudio.com/959adb23-f323-4d52-8203-ff34e5cbeefa/_packaging/marlinpi/pypi/simple --force-reinstall