#!/usr/bin/env bash

set -ex

# cleanup

rm -fr precice-run
rm -fr micro/micro-runs/*
rm -fr macro/visualization

# run

( cd macro && . .venv/bin/activate && python3 macro_porepy_3D.py > porepy.log 2>&1) &

( cd micro && . ~/Software/others/venvs/common/bin/activate && micro-manager-precice micro-manager-config.json > micro.log 2>&1)
