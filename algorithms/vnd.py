from typing import List, Callable, Tuple
import copy
import random
import time

def calculate_solution_value(solution: List[List[int]], 
                            release_times: List[int], 
                            processing_times: List[int], 
                            waiting_times: List[List[int]], 
                            penalties: List[int]) -> float:
    """
    Calculate the total penalty value of a solution.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Total penalty value
    """
    total_penalty = 0
    
    for runway in solution:
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time for this flight
            if prev_flight is None:
                # First flight on runway
                start_time = max(current_time, release_times[flight])
            else:
                # Consider waiting time between flights
                wait_time = waiting_times[prev_flight][flight]
                earliest_possible_time = current_time + wait_time
                start_time = max(earliest_possible_time, release_times[flight])
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            total_penalty += flight_penalty
            
            # Update current time and previous flight
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    return total_penalty

def calculate_incremental_change(original_solution: List[List[int]],
                                modified_solution: List[List[int]],
                                modified_runways: List[int],
                                release_times: List[int],
                                processing_times: List[int],
                                waiting_times: List[List[int]],
                                penalties: List[int]) -> float:
    """
    Calculate the change in solution value for only the modified runways.
    
    Args:
        original_solution: Original assignment of flights to runways
        modified_solution: Modified assignment of flights to runways
        modified_runways: Indices of runways that were modified
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Change in solution value (new_value - original_value)
    """
    # Calculate penalty for modified runways in original solution
    original_penalty = 0
    for runway_idx in modified_runways:
        if runway_idx >= len(original_solution):
            continue
            
        runway = original_solution[runway_idx]
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                wait_time = waiting_times[prev_flight][flight]
                earliest_possible_time = current_time + wait_time
                start_time = max(earliest_possible_time, release_times[flight])
            
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            original_penalty += flight_penalty
            
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    # Calculate penalty for modified runways in new solution
    new_penalty = 0
    for runway_idx in modified_runways:
        if runway_idx >= len(modified_solution):
            continue
            
        runway = modified_solution[runway_idx]
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                wait_time = waiting_times[prev_flight][flight]
                earliest_possible_time = current_time + wait_time
                start_time = max(earliest_possible_time, release_times[flight])
            
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            new_penalty += flight_penalty
            
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    # Return the change in penalty
    return new_penalty - original_penalty

def vnd_algorithm(initial_solution: List[List[int]],
                 release_times: List[int],
                 processing_times: List[int],
                 waiting_times: List[List[int]],
                 penalties: List[int],
                 neighborhood_functions: List[Callable],
                 max_iterations: int = 1000,
                 time_limit: float = 10.0) -> List[List[int]]:
    """
    Optimized Variable Neighborhood Descent algorithm for the runway scheduling problem.
    
    Args:
        initial_solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        neighborhood_functions: List of neighborhood functions to use
        max_iterations: Maximum number of iterations
        time_limit: Maximum time in seconds
        
    Returns:
        Improved solution
    """
    if not neighborhood_functions:
        return initial_solution
    
    current_solution = copy.deepcopy(initial_solution)
    current_value = calculate_solution_value(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    iteration = 0
    k = 0  # Index of the current neighborhood
    start_time = time.time()
    
    no_improvement_count = 0
    max_no_improvement = 5  # Allow 5 cycles through all neighborhoods without improvement
    
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value
    
    # Add a time-based intensification
    time_check_interval = 50  # Check time every 50 iterations
    
    while iteration < max_iterations and no_improvement_count < max_no_improvement:
        # Check time limit periodically to reduce overhead
        if iteration % time_check_interval == 0 and time.time() - start_time > time_limit:
            break
            
        # Get the current neighborhood function
        neighborhood_function = neighborhood_functions[k]
        
        # Generate a neighbor solution using the current neighborhood
        neighbor = neighborhood_function(
            current_solution, release_times, processing_times, waiting_times, penalties
        )
        
        # Get modified runways for incremental evaluation
        modified_runways = []
        for i in range(min(len(current_solution), len(neighbor))):
            if current_solution[i] != neighbor[i]:
                modified_runways.append(i)
        
        # If solution structure changed (e.g., new runway added), do full evaluation
        if len(current_solution) != len(neighbor):
            neighbor_value = calculate_solution_value(
                neighbor, release_times, processing_times, waiting_times, penalties
            )
        else:
            # Incremental evaluation - calculate change and update value
            value_change = calculate_incremental_change(
                current_solution, neighbor, modified_runways,
                release_times, processing_times, waiting_times, penalties
            )
            neighbor_value = current_value + value_change
        
        # If the neighbor improves the solution, accept it and reset k
        if neighbor_value < current_value:
            current_solution = neighbor  # No need for deepcopy since neighborhood function already creates copy
            current_value = neighbor_value
            k = 0  # Reset to the first neighborhood
            no_improvement_count = 0
            
            # Update best solution if current is better
            if current_value < best_value:
                best_solution = copy.deepcopy(current_solution)
                best_value = current_value
        else:
            # Move to the next neighborhood
            k += 1
            
            # If we've tried all neighborhoods without improvement
            if k >= len(neighborhood_functions):
                k = 0
                no_improvement_count += 1
        
        iteration += 1
    
    # Check if we spent less than half the time limit and could make an intensification phase
    elapsed_time = time.time() - start_time
    if elapsed_time < time_limit / 2 and iteration < max_iterations / 2:
        # Perform intensification by trying larger neighborhood moves
        remaining_time = time_limit - elapsed_time
        remaining_iterations = max_iterations - iteration
        
        for _ in range(min(100, remaining_iterations)):
            if time.time() - start_time > time_limit:
                break
                
            # Apply multiple neighborhood moves to escape local optima
            temp_solution = copy.deepcopy(best_solution)
            for func in neighborhood_functions:
                temp_solution = func(temp_solution, release_times, processing_times, waiting_times, penalties)
            
            temp_value = calculate_solution_value(
                temp_solution, release_times, processing_times, waiting_times, penalties
            )
            
            if temp_value < best_value:
                best_solution = temp_solution
                best_value = temp_value
    
    return best_solution

def first_improvement_vnd(initial_solution: List[List[int]],
                         release_times: List[int],
                         processing_times: List[int],
                         waiting_times: List[List[int]],
                         penalties: List[int],
                         neighborhood_functions: List[Callable],
                         max_iterations: int = 1000,
                         time_limit: float = 5.0,
                         max_neighbors_per_iter: int = 50) -> List[List[int]]:
    """
    Optimized Variable Neighborhood Descent with first improvement strategy.
    
    Args:
        initial_solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        neighborhood_functions: List of neighborhood functions to use
        max_iterations: Maximum number of iterations
        time_limit: Maximum time in seconds
        max_neighbors_per_iter: Maximum number of neighbors to check per iteration
        
    Returns:
        Improved solution
    """
    if not neighborhood_functions:
        return initial_solution
    
    current_solution = copy.deepcopy(initial_solution)
    current_value = calculate_solution_value(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value
    
    iteration = 0
    start_time = time.time()
    neighborhood_index = 0
    
    stagnation_count = 0
    max_stagnation = 5  # Allow 5 iterations without improvement before intensification
    
    # Precompute a lookup table for waiting times for faster access
    waiting_time_lookup = {}
    num_flights = len(release_times)
    for i in range(num_flights):
        for j in range(num_flights):
            waiting_time_lookup[(i, j)] = waiting_times[i][j]
    
    while iteration < max_iterations and stagnation_count < max_stagnation:
        if iteration % 20 == 0 and time.time() - start_time > time_limit:
            break
            
        improvement_found = False
        
        # Try each neighborhood in sequence, cycling through them
        for _ in range(len(neighborhood_functions)):  # Try each neighborhood once per iteration
            neighborhood_function = neighborhood_functions[neighborhood_index]
            neighborhood_index = (neighborhood_index + 1) % len(neighborhood_functions)
            
            # Generate and check multiple neighbors, accepting first improvement
            for _ in range(max_neighbors_per_iter):
                if time.time() - start_time > time_limit:
                    break
                    
                neighbor = neighborhood_function(
                    current_solution, release_times, processing_times, waiting_times, penalties
                )
                
                # Get modified runways for incremental evaluation
                modified_runways = []
                for i in range(min(len(current_solution), len(neighbor))):
                    if current_solution[i] != neighbor[i]:
                        modified_runways.append(i)
                
                # If solution structure changed, do full evaluation
                if len(current_solution) != len(neighbor):
                    neighbor_value = calculate_solution_value(
                        neighbor, release_times, processing_times, waiting_times, penalties
                    )
                else:
                    # Incremental evaluation
                    value_change = calculate_incremental_change(
                        current_solution, neighbor, modified_runways,
                        release_times, processing_times, waiting_times, penalties
                    )
                    neighbor_value = current_value + value_change
                
                if neighbor_value < current_value:
                    current_solution = neighbor
                    current_value = neighbor_value
                    improvement_found = True
                    
                    # Update best solution if improved
                    if current_value < best_value:
                        best_solution = copy.deepcopy(current_solution)
                        best_value = current_value
                    
                    break  # Found improvement, break inner loop
            
            if improvement_found:
                break  # Found improvement, break outer loop
        
        # If no improvement was found in any neighborhood
        if not improvement_found:
            stagnation_count += 1
            
            # Apply perturbation if stagnated
            if stagnation_count >= 2:
                # Generate a more diversified solution by applying multiple random moves
                perturbed_solution = copy.deepcopy(current_solution)
                for _ in range(3):  # Apply 3 random neighborhood functions
                    func = random.choice(neighborhood_functions)
                    perturbed_solution = func(perturbed_solution, release_times, processing_times, waiting_times, penalties)
                
                perturbed_value = calculate_solution_value(
                    perturbed_solution, release_times, processing_times, waiting_times, penalties
                )
                
                # Accept perturbation with 50% probability even if worse
                if perturbed_value < current_value or random.random() < 0.5:
                    current_solution = perturbed_solution
                    current_value = perturbed_value
                    stagnation_count = 0
        else:
            stagnation_count = 0
        
        iteration += 1
    
    # Perform final intensification on the best solution found
    elapsed_time = time.time() - start_time
    if elapsed_time < time_limit / 2:
        for _ in range(100):  # Try to improve the best solution further
            if time.time() - start_time > time_limit:
                break
                
            func = random.choice(neighborhood_functions)
            intensified = func(best_solution, release_times, processing_times, waiting_times, penalties)
            intensified_value = calculate_solution_value(
                intensified, release_times, processing_times, waiting_times, penalties
            )
            
            if intensified_value < best_value:
                best_solution = intensified
                best_value = intensified_value
    
    return best_solution
