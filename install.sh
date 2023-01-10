#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv tf-env

./tf-env/bin/python3.7 -m pip install --upgrade pip
# ./tf-env/bin/python3.7 pip install --force-reinstall pip==19


./tf-env/bin/pip install -r requirements.txt
source ./tf-env/bin/activate