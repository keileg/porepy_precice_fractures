#!/usr/bin/env bash

set -ex

python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

hash blockMesh
hash simpleFoam
hash micro-manager-precice

python3 -c "import micro.micro"

( cd macro && python3 macro.py ) &

( cd micro && micro-manager-precice micro-manager-config.json )
