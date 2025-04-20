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

def run_tests(test_dir: str = "tests", max_iterations: int = 1000) -> pd.DataFrame:
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".txt") and f.startswith("n")]
    expected_results = load_expected_results()
    
    results = []
    
    for test_file in sorted(test_files):
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
        
        # Determine time limit based on problem size
        if num_flights <= 10:
            time_limit = 5.0  # Small instances
        elif num_flights <= 20:
            time_limit = 10.0  # Medium instances
        else:
            time_limit = 20.0  # Large instances
        
        # Test different algorithms
        for alg_name, alg_func in [
            ("Earliest Ready Time", earliest_ready_time),
            ("Least Penalty", least_penalty),
            ("Combined Heuristic", combined_heuristic)
        ]:
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
            
            # Run with VND improvement
            print(f"  Running VND for {alg_name}...")
            start_time = time.time()
            improved_solution = vnd_algorithm(
                initial_solution,
                release_times,
                processing_times,
                waiting_times,
                penalties,
                neighborhood_functions,
                max_iterations,
                time_limit=time_limit
            )
            vnd_time = time.time() - start_time
            
            vnd_value = calculate_solution_value(
                improved_solution, release_times, processing_times, waiting_times, penalties
            )
            
            # Calculate gap
            vnd_gap = calculate_gap(vnd_value, expected_value) if expected_value != float('inf') else float('inf')
            
            # Store results for VND
            results.append({
                "Instance": instance_name,
                "Algorithm": f"{alg_name} + VND",
                "VND": True,
                "Solution Value": vnd_value,
                "Execution Time (s)": vnd_time,
                "Gap (%)": vnd_gap,
                "Expected Value": expected_value
            })
            
            # Run with first improvement VND
            print(f"  Running First Improvement VND for {alg_name}...")
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
            
            # Try with combined approach (shake solution + VND) for larger instances
            if num_flights > 20:
                print(f"  Running Combined Approach for {alg_name}...")
                start_time = time.time()
                
                # First shake the initial solution
                shaken_solution = shake_solution(
                    initial_solution,
                    release_times,
                    processing_times,
                    waiting_times,
                    penalties,
                    intensity=5  # Stronger perturbation for larger instances
                )
                
                # Apply VND to the shaken solution
                combined_solution = first_improvement_vnd(
                    shaken_solution,
                    release_times,
                    processing_times,
                    waiting_times,
                    penalties,
                    neighborhood_functions,
                    max_iterations,
                    time_limit=time_limit
                )
                
                combined_time = time.time() - start_time
                
                combined_value = calculate_solution_value(
                    combined_solution, release_times, processing_times, waiting_times, penalties
                )
                
                # Calculate gap
                combined_gap = calculate_gap(combined_value, expected_value) if expected_value != float('inf') else float('inf')
                
                # Store results for combined approach
                results.append({
                    "Instance": instance_name,
                    "Algorithm": f"{alg_name} + Shake + VND",
                    "VND": True,
                    "Solution Value": combined_value,
                    "Execution Time (s)": combined_time,
                    "Gap (%)": combined_gap,
                    "Expected Value": expected_value
                })
    
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
    print("Starting test runner...")
    results_df = run_tests()
    
    # Save results to CSV
    results_df.to_csv("test_results.csv", index=False)
    print("Results saved to test_results.csv")
    
    # Display summary
    summary = display_results_summary(results_df)
    print(summary)
    
    # Save summary to file
    with open("test_results_summary.txt", "w") as f:
        f.write(summary)
    print("Summary saved to test_results_summary.txt") 