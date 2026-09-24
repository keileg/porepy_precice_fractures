#!/usr/bin/env bash

set -euo pipefail

results_dir="./2d_flow_test_results"
bc_file="./macro/shared_flow.py"
macro_pid=""

mkdir -p "$results_dir"

if (( $# > 0 )); then
    pressure_differences=("$@")
else
    pressure_differences=(1250 1500)
    # pressure_differences=(50)
fi

run_case() {
    local pressure_difference="$1"
    local solve_flow_value="$2"
    local case_name="$3"

    rm -rf "./precice-run"
    rm -rf "./micro/micro-runs"
    mkdir -p "./micro/micro-runs"
    rm -f "./micro/*.csv"

    (
        cd "./macro"
        . .venv/bin/activate
        python3 2D.py > "porepy_${case_name}_dp_${pressure_difference}.log" 2>&1
    ) &
    macro_pid=$!

    (
        cd "./micro"
        . ../macro/.venv/bin/activate
        solve_flow="$solve_flow_value" \
        solve_scalar=0 \
        micro-manager-precice micro-manager-config.json \
            > "micro_${case_name}_dp_${pressure_difference}.log" 2>&1
    )

    wait "$macro_pid"
    macro_pid=""

    cp \
        "./macro/outlet_flux_2D.csv" \
        "$results_dir/outlet_flux_2D_dp_${pressure_difference}_${case_name}.csv"
}

for pressure_difference in "${pressure_differences[@]}"; do
    sed -E -i \
        's@(values\[domain_sides\.west\] = self\.units\.convert_units\()[^,]+(, "Pa"\))@\1'"$pressure_difference"'\2@' \
        "$bc_file"

    run_case "$pressure_difference" 0 analytical
    run_case "$pressure_difference" 1 coupling
done

(
    cp "./macro/plot_pressures.py" "$results_dir/plot_pressures.py"
    cd "$results_dir"
    . ../macro/.venv/bin/activate
    python3 plot_pressures.py
)
