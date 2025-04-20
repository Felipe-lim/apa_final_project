import os
import time
import pandas as pd
from io import StringIO
from typing import Dict, List, Tuple

from algorithms.greedy import earliest_ready_time, least_penalty, combined_heuristic
from algorithms.vnd import vnd_algorithm, first_improvement_vnd
from algorithms.neighborhood import swap_flights, move_flight, swap_flights_between_runways, reinsert_flight, reinsert_delayed_flight, optimize_runway_balance, shake_solution
from utils.file_handler import parse_input_file
from utils.metrics import calculate_solution_value, calculate_gap

def load_expected_results() -> Dict[str, float]:
    results = {}
    with open("tests/test_result_expected", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or ":" not in line and not any(c.isdigit() for c in line):
                continue
                
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("n") and any(c.isdigit() for c in parts[0]):
                instance_name = parts[0].strip().rstrip(":")
                value = float(parts[1].strip().rstrip("(opt)").rstrip("(LB)"))
                results[instance_name] = value
    
    return results

def run_quick_test(test_dir: str = "tests", max_iterations: int = 500) -> pd.DataFrame:
    # Select a small subset of test files
    test_files = ["n3m10A.txt", "n3m20A.txt", "n3m40A.txt"] 
    expected_results = load_expected_results()
    
    results = []
    
    for test_file in test_files:
        instance_name = test_file.split(".")[0]
        expected_value = expected_results.get(instance_name, float('inf'))
        
        print(f"Running test: {instance_name}")
        
        # Load test instance
        with open(os.path.join(test_dir, test_file), "r") as f:
            try:
                num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(f)
            except Exception as e:
                print(f"Error parsing file {test_file}: {e}")
                continue
        
        # Define neighborhood functions for VND with all available operations
        neighborhood_functions = [
            swap_flights,
            move_flight, 
            swap_flights_between_runways,
            reinsert_flight,
            reinsert_delayed_flight,
            optimize_runway_balance
        ]
        
        # Small fixed time limit for quick testing
        time_limit = 2.0
        
        # Test our best algorithm - Combined Heuristic with First Improvement VND
        alg_name = "Combined Heuristic"
        alg_func = combined_heuristic
        
        # Run constructive algorithm
        start_time = time.time()
        initial_solution = alg_func(release_times, processing_times, waiting_times, penalties, num_runways)
        constructive_time = time.time() - start_time
        
        constructive_value = calculate_solution_value(
            initial_solution, release_times, processing_times, waiting_times, penalties
        )
        
        # Calculate gap
        constructive_gap = calculate_gap(constructive_value, expected_value) if expected_value != float('inf') else float('inf')
        
        # Store results for constructive algorithm
        results.append({
            "Instance": instance_name,
            "Algorithm": alg_name,
            "VND": False,
            "Solution Value": constructive_value,
            "Execution Time (s)": constructive_time,
            "Gap (%)": constructive_gap,
            "Expected Value": expected_value
        })
        
        # Run with first improvement VND
        print(f"  Running First Improvement VND...")
        start_time = time.time()
        first_imp_solution = first_improvement_vnd(
            initial_solution,
            release_times,
            processing_times,
            waiting_times,
            penalties,
            neighborhood_functions,
            max_iterations,
            time_limit=time_limit
        )
        first_imp_time = time.time() - start_time
        
        first_imp_value = calculate_solution_value(
            first_imp_solution, release_times, processing_times, waiting_times, penalties
        )
        
        # Calculate gap
        first_imp_gap = calculate_gap(first_imp_value, expected_value) if expected_value != float('inf') else float('inf')
        
        # Store results for first improvement VND
        results.append({
            "Instance": instance_name,
            "Algorithm": f"{alg_name} + First Improvement VND",
            "VND": True,
            "Solution Value": first_imp_value,
            "Execution Time (s)": first_imp_time,
            "Gap (%)": first_imp_gap,
            "Expected Value": expected_value
        })
        
        # Add running time report to verify actual execution time
        print(f"  Execution time: {first_imp_time:.2f} seconds")
        print(f"  Solution value: {first_imp_value} (GAP: {first_imp_gap:.2f}%)")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

def display_results_summary(results_df: pd.DataFrame) -> str:
    output = StringIO()
    
    # Summary by instance
    print("\n==== Results Summary by Instance ====", file=output)
    instance_summary = results_df.groupby("Instance")[["Gap (%)", "Execution Time (s)"]].min()
    print(instance_summary.sort_values("Gap (%)"), file=output)
    
    # Summary by algorithm
    print("\n==== Results Summary by Algorithm ====", file=output)
    alg_summary = results_df.groupby("Algorithm")[["Gap (%)", "Execution Time (s)", "Solution Value"]].mean()
    print(alg_summary.sort_values("Gap (%)"), file=output)
    
    # Best algorithm for each instance
    print("\n==== Best Algorithm for Each Instance ====", file=output)
    for instance in sorted(results_df["Instance"].unique()):
        instance_df = results_df[results_df["Instance"] == instance]
        best_row = instance_df.loc[instance_df["Solution Value"].idxmin()]
        print(f"{instance}: {best_row['Algorithm']} - Value: {best_row['Solution Value']} - Gap: {best_row['Gap (%)']}%", file=output)
    
    return output.getvalue()

if __name__ == "__main__":
    print("Starting quick test runner...")
    results_df = run_quick_test()
    
    # Display summary
    summary = display_results_summary(results_df)
    print(summary)
    
    print("Quick test completed!") 