import os
import time
import copy
import random
from typing import List, Dict, Callable

from algorithms.greedy import earliest_ready_time, least_penalty, combined_heuristic
from algorithms.vnd import vnd_algorithm, first_improvement_vnd
from algorithms.neighborhood import (
    swap_flights, move_flight, swap_flights_between_runways, 
    reinsert_flight, reinsert_delayed_flight, optimize_runway_balance, 
    shake_solution
)
from utils.file_handler import parse_input_file
from utils.metrics import calculate_solution_value, calculate_gap
from utils.visualization import plot_runway_schedule
import matplotlib.pyplot as plt

# Target instance
TARGET_INSTANCE = "n3m40D"
TARGET_FILE = f"tests/{TARGET_INSTANCE}.txt"
EXPECTED_VALUE = 263  # From test_result_expected

# Load the instance
with open(TARGET_FILE, "r") as f:
    num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(f)

print(f"Instance {TARGET_INSTANCE} loaded: {num_flights} flights, {num_runways} runways")

# Define various neighborhood combinations to test
NEIGHBORHOOD_SETS = [
    # Basic neighborhoods
    [swap_flights, move_flight, swap_flights_between_runways, reinsert_flight],
    
    # All neighborhoods
    [swap_flights, move_flight, swap_flights_between_runways, reinsert_flight, 
     reinsert_delayed_flight, optimize_runway_balance],
    
    # Focused on flight movement
    [move_flight, reinsert_flight, reinsert_delayed_flight],
    
    # Prioritize delay reduction
    [reinsert_delayed_flight, optimize_runway_balance, swap_flights_between_runways],
]

# Helper function to print solution details
def print_solution_details(solution, heading="Solution"):
    value = calculate_solution_value(solution, release_times, processing_times, waiting_times, penalties)
    gap = calculate_gap(value, EXPECTED_VALUE)
    
    print(f"\n{heading}:")
    print(f"  Value: {value}")
    print(f"  Gap: {gap:.2f}%")
    print(f"  Flights per runway: {[len(runway) for runway in solution]}")
    
    return value, gap

# Multi-start strategy: Generate multiple initial solutions
def multi_start_strategy(num_starts=5, time_limit=60):
    print("\n=== Running Multi-Start Strategy ===")
    best_solution = None
    best_value = float('inf')
    
    for i in range(num_starts):
        print(f"\nStart {i+1}/{num_starts}")
        
        # Use a randomized greedy function for diversification
        constructive_algorithm = random.choice([earliest_ready_time, least_penalty, combined_heuristic])
        print(f"Using constructive algorithm: {constructive_algorithm.__name__}")
        
        # Generate initial solution
        initial_solution = constructive_algorithm(release_times, processing_times, waiting_times, penalties, num_runways)
        
        # Apply small perturbation
        initial_solution = shake_solution(initial_solution, release_times, processing_times, waiting_times, penalties, intensity=2)
        
        # Improve with VND
        neighborhood_functions = random.choice(NEIGHBORHOOD_SETS)
        print(f"Using {len(neighborhood_functions)} neighborhood functions")
        
        start_time = time.time()
        improved_solution = first_improvement_vnd(
            initial_solution,
            release_times,
            processing_times,
            waiting_times,
            penalties,
            neighborhood_functions,
            max_iterations=2000,
            time_limit=time_limit/num_starts,
            max_neighbors_per_iter=100
        )
        elapsed = time.time() - start_time
        
        # Evaluate solution
        value = calculate_solution_value(improved_solution, release_times, processing_times, waiting_times, penalties)
        gap = calculate_gap(value, EXPECTED_VALUE)
        print(f"Solution value: {value}, Gap: {gap:.2f}%, Time: {elapsed:.2f}s")
        
        if value < best_value:
            best_solution = copy.deepcopy(improved_solution)
            best_value = value
            print(f"New best solution found!")
    
    print("\nBest solution from multi-start strategy:")
    print_solution_details(best_solution, "Best Multi-Start Solution")
    return best_solution

# Iterative improvement strategy
def iterative_improvement_strategy(solution, max_iterations=10, time_limit=120):
    print("\n=== Running Iterative Improvement Strategy ===")
    
    best_solution = copy.deepcopy(solution)
    best_value = calculate_solution_value(best_solution, release_times, processing_times, waiting_times, penalties)
    
    for i in range(max_iterations):
        print(f"\nIteration {i+1}/{max_iterations}")
        
        # Step 1: Apply perturbation to escape local optima
        intensity = random.randint(2, 5)  # Random intensity
        perturbed = shake_solution(best_solution, release_times, processing_times, waiting_times, penalties, intensity)
        
        # Step 2: Apply specialized optimization focusing on high-penalty flights
        optimized = reinsert_delayed_flight(perturbed, release_times, processing_times, waiting_times, penalties)
        optimized = optimize_runway_balance(optimized, release_times, processing_times, waiting_times, penalties)
        
        # Step 3: Apply VND with different neighborhood sets
        neighborhood_functions = NEIGHBORHOOD_SETS[i % len(NEIGHBORHOOD_SETS)]
        print(f"Using neighborhood set {(i % len(NEIGHBORHOOD_SETS)) + 1}")
        
        start_time = time.time()
        improved = first_improvement_vnd(
            optimized,
            release_times,
            processing_times,
            waiting_times,
            penalties,
            neighborhood_functions,
            max_iterations=1500,
            time_limit=time_limit/max_iterations,
            max_neighbors_per_iter=100
        )
        elapsed = time.time() - start_time
        
        # Evaluate current solution
        current_value = calculate_solution_value(improved, release_times, processing_times, waiting_times, penalties)
        gap = calculate_gap(current_value, EXPECTED_VALUE)
        print(f"Solution value: {current_value}, Gap: {gap:.2f}%, Time: {elapsed:.2f}s")
        
        # Update best solution if improved
        if current_value < best_value:
            best_solution = copy.deepcopy(improved)
            best_value = current_value
            print(f"New best solution found!")
            
        # Early termination if gap is very small
        if gap < 5.0:
            print(f"Gap is below 5%, stopping early")
            break
    
    print("\nBest solution from iterative improvement:")
    print_solution_details(best_solution, "Best Iterative Solution")
    return best_solution

# Intensive local search on highest penalty flights
def intensive_local_search(solution, max_time=60):
    print("\n=== Running Intensive Local Search on High-Penalty Flights ===")
    
    # Calculate penalties for all flights in the current solution
    flight_penalties = {}
    flight_delays = {}
    flight_locations = {}  # (runway_idx, position)
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for pos, flight in enumerate(runway):
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            delay = max(0, start_time - release_times[flight])
            flight_penalties[flight] = delay * penalties[flight]
            flight_delays[flight] = delay
            flight_locations[flight] = (runway_idx, pos)
            
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    # Sort flights by penalty (descending)
    high_penalty_flights = sorted(flight_penalties.keys(), key=lambda f: flight_penalties[f], reverse=True)
    
    # Take top 25% of flights with highest penalties
    num_flights_to_consider = max(len(high_penalty_flights) // 4, 5)
    target_flights = high_penalty_flights[:num_flights_to_consider]
    
    print(f"Focusing on {len(target_flights)} flights with highest penalties")
    for i, flight in enumerate(target_flights[:5]):  # Print details for top 5
        print(f"  Flight {flight}: Penalty {flight_penalties[flight]}, Delay {flight_delays[flight]}")
    
    best_solution = copy.deepcopy(solution)
    best_value = calculate_solution_value(best_solution, release_times, processing_times, waiting_times, penalties)
    
    start_time = time.time()
    elapsed = 0
    iteration = 0
    
    while elapsed < max_time:
        iteration += 1
        print(f"\nIteration {iteration}, elapsed time: {elapsed:.2f}s")
        
        # Make a working copy
        current_solution = copy.deepcopy(best_solution)
        improved = False
        
        # Try to optimize each high-penalty flight
        for flight in target_flights:
            # Skip if flight no longer exists in solution (due to previous changes)
            found = False
            flight_runway_idx = None
            flight_pos = None
            
            for runway_idx, runway in enumerate(current_solution):
                if flight in runway:
                    found = True
                    flight_runway_idx = runway_idx
                    flight_pos = runway.index(flight)
                    break
            
            if not found:
                continue
            
            # Try all possible positions in all runways
            best_move_value = float('inf')
            best_move = None
            
            # Remove flight from current position
            current_runway = current_solution[flight_runway_idx]
            flight_obj = current_runway.pop(flight_pos)
            
            for target_runway_idx in range(len(current_solution)):
                target_runway = current_solution[target_runway_idx]
                
                for insert_pos in range(len(target_runway) + 1):
                    # Insert flight temporarily
                    target_runway.insert(insert_pos, flight_obj)
                    
                    # Calculate new solution value
                    new_value = calculate_solution_value(current_solution, release_times, processing_times, 
                                                       waiting_times, penalties)
                    
                    # Check if this is better
                    if new_value < best_move_value:
                        best_move_value = new_value
                        best_move = (target_runway_idx, insert_pos)
                    
                    # Remove the flight
                    target_runway.pop(insert_pos)
            
            # If we found a better position, apply it
            if best_move and best_move_value < best_value:
                target_runway_idx, insert_pos = best_move
                current_solution[target_runway_idx].insert(insert_pos, flight_obj)
                improved = True
                best_value = best_move_value
                best_solution = copy.deepcopy(current_solution)
                print(f"  Improved solution by relocating flight {flight}")
            else:
                # Put the flight back where it was
                current_solution[flight_runway_idx].insert(flight_pos, flight_obj)
        
        # Apply general local search if no improvements from individual flight moves
        if not improved:
            print("  No improvements from individual flight moves, applying general local search")
            neighborhood_functions = NEIGHBORHOOD_SETS[iteration % len(NEIGHBORHOOD_SETS)]
            
            improved_solution = first_improvement_vnd(
                current_solution,
                release_times,
                processing_times,
                waiting_times,
                penalties,
                neighborhood_functions,
                max_iterations=200,
                time_limit=5.0,  # Short time limit for each attempt
                max_neighbors_per_iter=50
            )
            
            new_value = calculate_solution_value(improved_solution, release_times, processing_times, 
                                               waiting_times, penalties)
            
            if new_value < best_value:
                best_solution = copy.deepcopy(improved_solution)
                best_value = new_value
                print(f"  Improved solution via general local search")
        
        elapsed = time.time() - start_time
        
        # Calculate current gap
        gap = calculate_gap(best_value, EXPECTED_VALUE)
        print(f"Current best: Value {best_value}, Gap {gap:.2f}%")
        
        # Early termination if we're close to optimal
        if gap < 5.0:
            print("Gap is below 5%, stopping early")
            break
    
    print("\nBest solution from intensive local search:")
    print_solution_details(best_solution, "Best Intensive Local Search Solution")
    return best_solution

# Main optimization process
def main():
    print(f"Starting optimization for {TARGET_INSTANCE}")
    print(f"Target value: {EXPECTED_VALUE}")
    
    # Step 1: Generate initial solutions using different constructive algorithms
    print("\n=== Generating Initial Solutions ===")
    constructive_algorithms = [
        ("Earliest Ready Time", earliest_ready_time),
        ("Least Penalty", least_penalty),
        ("Combined Heuristic", combined_heuristic)
    ]
    
    best_constructive = None
    best_constructive_value = float('inf')
    
    for name, algorithm in constructive_algorithms:
        print(f"\nTrying {name}...")
        solution = algorithm(release_times, processing_times, waiting_times, penalties, num_runways)
        value, gap = print_solution_details(solution, f"{name} Solution")
        
        if value < best_constructive_value:
            best_constructive = copy.deepcopy(solution)
            best_constructive_value = value
            print(f"New best constructive solution!")
    
    # Step 2: Apply multi-start strategy
    best_multi_start = multi_start_strategy(num_starts=5, time_limit=60)
    
    # Step 3: Apply iterative improvement on the best solution so far
    current_best = best_multi_start
    current_best_value = calculate_solution_value(current_best, release_times, processing_times, 
                                                waiting_times, penalties)
    
    if current_best_value < best_constructive_value:
        print("\nUsing multi-start solution for further improvement")
    else:
        current_best = best_constructive
        current_best_value = best_constructive_value
        print("\nUsing best constructive solution for further improvement")
    
    # Apply iterative improvement
    best_iterative = iterative_improvement_strategy(current_best, max_iterations=10, time_limit=120)
    
    # Step 4: Apply intensive local search on high-penalty flights
    best_intensive = intensive_local_search(best_iterative, max_time=90)
    
    # Final solution
    final_value, final_gap = print_solution_details(best_intensive, "Final Solution")
    
    # Visualize the final solution
    fig = plot_runway_schedule(best_intensive, release_times, processing_times, penalties, num_runways, waiting_times)
    plt.savefig(f"{TARGET_INSTANCE}_optimized_solution.png")
    plt.close(fig)
    
    # Write solution to file
    with open(f"{TARGET_INSTANCE}_optimized_solution.txt", "w") as f:
        f.write(f"{final_value}\n")
        for runway in best_intensive:
            # Convert to 1-based indices for output
            f.write(" ".join(str(flight + 1) for flight in runway) + "\n")
    
    print(f"\nOptimization complete! Final gap: {final_gap:.2f}%")
    print(f"Solution saved to {TARGET_INSTANCE}_optimized_solution.txt")
    print(f"Visualization saved to {TARGET_INSTANCE}_optimized_solution.png")

if __name__ == "__main__":
    main() 