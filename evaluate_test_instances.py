import time
import glob
import os
import re
import copy
import csv
import io # For CSV output
import math # For checking nan/inf

# Import GRASP construction
from algorithms.greedy import combined_heuristic, grasp_construction
from algorithms.vnd import vnd_algorithm
from algorithms.ils import iterated_local_search
from algorithms.neighborhood import (
    swap_flights, move_flight, swap_flights_between_runways, 
    reinsert_flight, reinsert_delayed_flight,
    shake_solution
)
from utils.file_handler import parse_input_file
from utils.metrics import calculate_solution_value, calculate_gap

# --- Expected Results (Parsed from tests/test_result_expected) ---
# Format: instance_name: (value, type)
EXPECTED_RESULTS = {
  "n3m10A": (7483, "opt"),
  "n3m10B": (1277, "opt"),
  "n3m10C": (2088, "opt"),
  "n3m10D": (322, "opt"),
  "n3m10E": (3343, "opt"),
  "n3m20A": (8280, "LB"), # Using the updated value
  "n3m20B": (1820, "LB"), # Using the updated value
  "n3m20C": (855, "LB"),
  "n3m20D": (4357, "opt"),
  "n3m20E": (3798, "opt"),
  "n3m40A": (112, "LB"),
  "n3m40B": (880, "LB"),
  "n3m40C": (1962, "LB"),
  "n3m40D": (263, "LB"),
  "n3m40E": (1192, "LB"),
  "n5m50A": (0, "LB"),
  "n5m50B": (0, "LB"),
  "n5m50C": (0, "LB"),
  "n5m50D": (0, "LB"),
  "n5m50E": (0, "LB"),
}

# --- Algorithm Execution Functions ---

def run_greedy(instance_data):
    """Runs the combined greedy heuristic."""
    rt, pt, wt, p, nr = instance_data
    start_time = time.time()
    schedule = combined_heuristic(rt, pt, wt, p, nr)
    penalty = calculate_solution_value(schedule, rt, pt, wt, p)
    exec_time = time.time() - start_time
    return penalty, exec_time

def run_vnd(instance_data, vnd_iter=1000, vnd_time=5.0):
    """Runs Greedy + VND."""
    rt, pt, wt, p, nr = instance_data
    start_time = time.time()
    # Initial Greedy Solution
    initial_schedule = combined_heuristic(rt, pt, wt, p, nr)
    # VND Neighborhoods
    neighborhoods = [
        swap_flights, move_flight, swap_flights_between_runways,
        reinsert_flight, reinsert_delayed_flight
    ]
    # Run VND
    final_schedule = vnd_algorithm(
        initial_schedule, rt, pt, wt, p, neighborhoods,
        max_iterations=vnd_iter, time_limit=vnd_time
    )
    final_penalty = calculate_solution_value(final_schedule, rt, pt, wt, p)
    exec_time = time.time() - start_time
    return final_penalty, exec_time

def run_ils(instance_data, ils_iter=10, ils_time=30.0, perturb_intensity=7, grasp_alpha=0.3, time_limit_ls=1.5):
    """Runs Iterated Local Search (with GRASP start)."""
    rt, pt, wt, p, nr = instance_data
    start_time = time.time()
    # Define neighborhoods for the inner local search (VND)
    vnd_neighborhoods = [
        swap_flights, move_flight, swap_flights_between_runways,
        reinsert_flight, reinsert_delayed_flight
    ]
    # Define GRASP constructor
    constructive_func = lambda r_t, p_t, w_t, pen, n_r: grasp_construction(r_t, p_t, w_t, pen, n_r, alpha=grasp_alpha)
    # Removed dynamic calculation, use parameter directly
    max_iterations_ls = 1000 # Keep inner iterations fixed

    # Run ILS
    best_schedule, final_penalty = iterated_local_search(
        rt, pt, wt, p, nr,
        constructive_heuristic=constructive_func,
        local_search=vnd_algorithm,
        perturbation=shake_solution,
        neighborhoods_for_ls=vnd_neighborhoods,
        max_iterations_ils=ils_iter,
        max_time_ils=ils_time,
        perturbation_intensity=perturb_intensity,
        max_iterations_ls=max_iterations_ls,
        time_limit_ls=time_limit_ls # Pass the fixed limit
    )
    exec_time = time.time() - start_time
    return final_penalty, exec_time

def run_grasp(instance_data, grasp_iter=10, grasp_alpha=0.3, vnd_iter=1000, vnd_time=5.0):
    """Runs GRASP metaheuristic (Construction + VND, repeated)."""
    rt, pt, wt, p, nr = instance_data
    start_time = time.time()
    best_overall_penalty = float('inf')
    best_overall_schedule = None

    # VND Neighborhoods (used inside loop)
    neighborhoods = [
        swap_flights, move_flight, swap_flights_between_runways,
        reinsert_flight, reinsert_delayed_flight
    ]

    for i in range(grasp_iter):
        # 1. GRASP Construction
        constructed_schedule = grasp_construction(rt, pt, wt, p, nr, alpha=grasp_alpha)
        
        # 2. Local Search (VND)
        improved_schedule = vnd_algorithm(
            constructed_schedule, rt, pt, wt, p, neighborhoods,
            max_iterations=vnd_iter, time_limit=vnd_time
        )
        current_penalty = calculate_solution_value(improved_schedule, rt, pt, wt, p)

        # Update best found solution
        if current_penalty < best_overall_penalty:
            best_overall_penalty = current_penalty
            # best_overall_schedule = copy.deepcopy(improved_schedule) # Keep if needed

    exec_time = time.time() - start_time
    # Handle case where no solution was found (shouldn't happen here)
    if best_overall_penalty == float('inf'):
        best_overall_penalty = None 

    return best_overall_penalty, exec_time

# --- Main Evaluation Loop --- 
if __name__ == "__main__":
    test_dir = "tests"
    instance_pattern = os.path.join(test_dir, "n*.txt")
    instance_files = sorted(glob.glob(instance_pattern))

    # --- Algorithm Parameters --- 
    # Updated parameters
    PARAMS = {
        'Greedy': {'iter': 1},
        'VND': {'iter': 1, 'vnd_iter': 1000, 'vnd_time': 5.0}, 
        'ILS': {'iter': 20, 'ils_time': 30.0, 'perturb': 10, 'alpha': 0.3, 'time_limit_ls': 1.5}, 
        'GRASP': {'iter': 20, 'grasp_alpha': 0.3, 'vnd_iter': 1000, 'vnd_time': 5.0}
    }
    ALGORITHMS_TO_RUN = list(PARAMS.keys())
    # --------------------------

    results_list = [] # Store results for CSV output

    if not instance_files:
        print(f"No instance files found in {test_dir} matching pattern {instance_pattern}")
    else:
        print(f"Found {len(instance_files)} instances in {test_dir}.")
        print("Running evaluations for Greedy, VND, ILS, GRASP...")
        print("Parameters:", PARAMS)
        print("-" * 60)

        for instance_path in instance_files:
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]
            print(f"Processing: {instance_name:<10} ... ", end="", flush=True)
            
            try:
                with open(instance_path, 'r') as f:
                  num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(f)
                instance_data = (release_times, processing_times, waiting_times, penalties, num_runways)
            except Exception as e:
                print(f"ERROR parsing {instance_name}: {e}")
                continue # Skip to next instance

            ref_value, ref_type = EXPECTED_RESULTS.get(instance_name, (None, None))

            # Run algorithms
            instance_run_results = {}
            run_error = False
            for algo_name in ALGORITHMS_TO_RUN:
                try:
                    penalty, ex_time = float('nan'), float('nan') # Default on error
                    if algo_name == 'Greedy':
                        penalty, ex_time = run_greedy(instance_data)
                    elif algo_name == 'VND':
                        penalty, ex_time = run_vnd(instance_data, vnd_iter=PARAMS['VND']['vnd_iter'], vnd_time=PARAMS['VND']['vnd_time'])
                    elif algo_name == 'ILS':
                        penalty, ex_time = run_ils(instance_data, ils_iter=PARAMS['ILS']['iter'], ils_time=PARAMS['ILS']['ils_time'], 
                                                 perturb_intensity=PARAMS['ILS']['perturb'], grasp_alpha=PARAMS['ILS']['alpha'], 
                                                 time_limit_ls=PARAMS['ILS']['time_limit_ls'])
                    elif algo_name == 'GRASP':
                        penalty, ex_time = run_grasp(instance_data, grasp_iter=PARAMS['GRASP']['iter'], grasp_alpha=PARAMS['GRASP']['grasp_alpha'], 
                                                  vnd_iter=PARAMS['GRASP']['vnd_iter'], vnd_time=PARAMS['GRASP']['vnd_time'])
                    
                    # Calculate GAP
                    gap = float('nan')
                    if penalty is not None and not math.isnan(penalty) and ref_value is not None:
                        gap = calculate_gap(penalty, ref_value)
                    
                    instance_run_results[algo_name] = {'Penalty': penalty, 'Time': ex_time, 'GAP': gap}
                    
                    # Add to overall list for final CSV
                    results_list.append({
                        'Instance': instance_name, 'Algorithm': algo_name, 'Penalty': penalty,
                        'Time': ex_time, 'Ref_Value': ref_value, 'Ref_Type': ref_type, 'GAP': gap
                    })

                except Exception as e:
                    print(f"ERROR running {algo_name}: {e}")
                    instance_run_results[algo_name] = {'Penalty': 'ERROR', 'Time': 'ERROR', 'GAP': 'ERROR'}
                    run_error = True
                    # Also add error marker to main list
                    results_list.append({
                        'Instance': instance_name, 'Algorithm': algo_name, 'Penalty': None,
                        'Time': None, 'Ref_Value': ref_value, 'Ref_Type': ref_type, 'GAP': None
                    })
            
            # Print formatted table for this instance
            print("-" * 50)
            print(f"  {'Algorithm':<10} {'Penalty':>12} {'Time (s)':>10} {'GAP (%)':>10}")
            print("-" * 50)
            for algo_name in ALGORITHMS_TO_RUN:
                 res = instance_run_results.get(algo_name, {})
                 
                 # Simplify formatting logic
                 penalty_val = res.get('Penalty', 'N/A')
                 time_val = res.get('Time', 'N/A')
                 gap_val = res.get('GAP')

                 pen_str = f"{penalty_val:>12.2f}" if isinstance(penalty_val, (int, float)) and not math.isnan(penalty_val) else f"{str(penalty_val):>12}"
                 time_str = f"{time_val:>10.4f}" if isinstance(time_val, (int, float)) and not math.isnan(time_val) else f"{str(time_val):>10}"
                 
                 gap_str = "N/A"
                 if isinstance(gap_val, (int, float)) and not math.isnan(gap_val):
                     if gap_val == float('inf'):
                         gap_str = "inf"
                     else:
                         gap_str = f"{gap_val:.2f}"
                 
                 print(f"  {algo_name:<10} {pen_str} {time_str} {gap_str:>10}") # Adjusted gap formatting
            print("-" * 50)
            print("=" * 60) # Separator for next instance

        print("-" * 60)
        print("Evaluation complete.")
        print("-" * 60)
        
        # --- Output results as CSV --- 
        if results_list:
            print("Final Results (CSV Format):")
            output = io.StringIO()
            fieldnames = ['Instance', 'Algorithm', 'Penalty', 'Time', 'Ref_Value', 'Ref_Type', 'GAP']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            # Filter out rows potentially added just for error markers if needed
            valid_results = [row for row in results_list if row['Penalty'] is not None or row['Time'] is not None] 
            writer.writerows(valid_results)
            
            print(output.getvalue())
        else:
            print("No results generated.")

    print("Script finished.") 