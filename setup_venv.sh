#!/bin/bash

VENV_DIR=./venv

if  [[ ! -d $VENV_DIR ]]
then
    #mkdir $VENV_DIR
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
pip install -r requirements.txt
deactivate