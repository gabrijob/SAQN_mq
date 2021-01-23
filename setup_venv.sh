#!/bin/bash

VENV_DIR = 'venv'

if  [[ ! -d "$VENV_DIR" ]]
then
    mkdir "$VENV_DIR"
fi

source ./venv/bin/activate
pip install -r requirements.txt
deactivate
