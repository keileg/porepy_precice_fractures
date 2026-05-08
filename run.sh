#!/usr/bin/env bash

set -ex

# setup venv
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

# check if all commands are available
hash blockMesh
hash simpleFoam
hash micro-manager-precice

# check if the micro simulation is importable
python3 -c "import micro.micro"

# cleanup

rm -fr precice-run

# run

( cd macro && python3 macro.py ) &

( cd micro && micro-manager-precice micro-manager-config.json )
