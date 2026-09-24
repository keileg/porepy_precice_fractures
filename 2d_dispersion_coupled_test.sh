#!/usr/bin/env bash

set -euo pipefail

results_dir="./2d_dispersion_test_results"
bc_file="./macro/shared_flow.py"
macro_pid=""

mkdir -p "$results_dir"

if (( $# > 0 )); then
    pressure_differences=("$@")
else
    pressure_differences=(2000)
fi

run_case() {
    local pressure_difference="$1"
    local solve_scalar_value="$2"
    local case_name="$3"

    rm -rf "./precice-run"
    rm -rf "./micro/micro-runs"
    mkdir -p "./micro/micro-runs"

    (
        cd "./macro"
        . .venv/bin/activate
        python3 2D.py > "porepy_${case_name}_dp_${pressure_difference}.log" 2>&1
    ) &
    macro_pid=$!

    (
        cd "./micro"
        . ../macro/.venv/bin/activate
        solve_flow=1 \
        solve_scalar="$solve_scalar_value" \
        micro-manager-precice micro-manager-config.json \
            > "micro_${case_name}_dp_${pressure_difference}.log" 2>&1
    )

    wait "$macro_pid"
    macro_pid=""

    cp \
        "./macro/BTC_2D.csv" \
        "$results_dir/BTC_2D_dp_${pressure_difference}_${case_name}.csv"
}

for pressure_difference in "${pressure_differences[@]}"; do
    sed -E -i \
        's@(values\[domain_sides\.west\] = self\.units\.convert_units\()[^,]+(, "Pa"\))@\1'"$pressure_difference"'\2@' \
        "$bc_file"

    run_case "$pressure_difference" 0 analytical
    run_case "$pressure_difference" 1 coupling
done

(
    # cp "./macro/plot_BTCs.py" "$results_dir/plot_BTCs.py"
    cd "$results_dir"
    . ../macro/.venv/bin/activate
    python3 plot_BTCs.py
)
