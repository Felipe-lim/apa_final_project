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

def vnd_algorithm(initial_solution: List[List[int]],
                 release_times: List[int],
                 processing_times: List[int],
                 waiting_times: List[List[int]],
                 penalties: List[int],
                 neighborhood_functions: List[Callable],
                 max_iterations: int = 100,
                 time_limit: float = 5.0) -> List[List[int]]:
    """
    Variable Neighborhood Descent algorithm for the runway scheduling problem.
    
    Args:
        initial_solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        neighborhood_functions: List of neighborhood functions to use
        max_iterations: Maximum number of iterations
        time_limit: Maximum execution time in seconds
        
    Returns:
        Improved solution
    """
    if not neighborhood_functions:
        return initial_solution
    
    start_time = time.time()
    current_solution = copy.deepcopy(initial_solution)
    current_value = calculate_solution_value(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value
    
    iteration = 0
    k = 0  # Index of the current neighborhood
    
    while k < len(neighborhood_functions) and iteration < max_iterations:
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
            
        # Get the current neighborhood function
        neighborhood_function = neighborhood_functions[k]
        
        # Generate a neighbor solution using the current neighborhood
        neighbor = neighborhood_function(
            current_solution, release_times, processing_times, waiting_times, penalties
        )
        
        # Evaluate the neighbor
        neighbor_value = calculate_solution_value(
            neighbor, release_times, processing_times, waiting_times, penalties
        )
        
        # If the neighbor improves the solution, accept it and reset k
        if neighbor_value < current_value:
            current_solution = copy.deepcopy(neighbor)
            current_value = neighbor_value
            
            # Update best solution if needed
            if current_value < best_value:
                best_solution = copy.deepcopy(current_solution)
                best_value = current_value
                
            k = 0  # Reset to the first neighborhood
        else:
            # Move to the next neighborhood
            k += 1
        
        iteration += 1
    
    return best_solution

def first_improvement_vnd(initial_solution: List[List[int]],
                         release_times: List[int],
                         processing_times: List[int],
                         waiting_times: List[List[int]],
                         penalties: List[int],
                         neighborhood_functions: List[Callable],
                         max_iterations: int = 100,
                         time_limit: float = 5.0,
                         neighbors_per_iteration: int = 50) -> List[List[int]]:
    """
    Variable Neighborhood Descent with first improvement strategy.
    
    Args:
        initial_solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        neighborhood_functions: List of neighborhood functions to use
        max_iterations: Maximum number of iterations
        time_limit: Maximum execution time in seconds
        neighbors_per_iteration: Number of neighbors to explore per iteration
        
    Returns:
        Improved solution
    """
    if not neighborhood_functions:
        return initial_solution
    
    start_time = time.time()
    current_solution = copy.deepcopy(initial_solution)
    current_value = calculate_solution_value(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value
    
    iteration = 0
    
    while iteration < max_iterations:
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
            
        improvement_found = False
        
        # Try each neighborhood in sequence
        for neighborhood_function in neighborhood_functions:
            # Generate multiple neighbors and accept the first improvement
            for _ in range(min(neighbors_per_iteration, max_iterations - iteration)):
                # Check time limit
                if time.time() - start_time >= time_limit:
                    break
                
                neighbor = neighborhood_function(
                    current_solution, release_times, processing_times, waiting_times, penalties
                )
                
                neighbor_value = calculate_solution_value(
                    neighbor, release_times, processing_times, waiting_times, penalties
                )
                
                if neighbor_value < current_value:
                    current_solution = copy.deepcopy(neighbor)
                    current_value = neighbor_value
                    
                    # Update best solution if needed
                    if current_value < best_value:
                        best_solution = copy.deepcopy(current_solution)
                        best_value = current_value
                        
                    improvement_found = True
                    break  # Found improvement, break inner loop
            
            if improvement_found:
                break  # Found improvement, break outer loop
        
        # If no improvement was found in any neighborhood, stop
        if not improvement_found:
            break
        
        iteration += 1
    
    return best_solution

def intensified_vnd(initial_solution: List[List[int]],
                   release_times: List[int],
                   processing_times: List[int],
                   waiting_times: List[List[int]],
                   penalties: List[int],
                   neighborhood_functions: List[Callable],
                   max_iterations: int = 500,
                   time_limit: float = 10.0) -> List[List[int]]:
    """
    Intensified VND that runs multiple iterations with different neighborhood orderings.
    
    Args:
        initial_solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        neighborhood_functions: List of neighborhood functions to use
        max_iterations: Maximum number of iterations
        time_limit: Maximum execution time in seconds
        
    Returns:
        Improved solution
    """
    if not neighborhood_functions:
        return initial_solution
    
    start_time = time.time()
    best_solution = copy.deepcopy(initial_solution)
    best_value = calculate_solution_value(
        best_solution, release_times, processing_times, waiting_times, penalties
    )
    
    # Try different orderings of neighborhoods
    neighborhood_orderings = []
    
    # Original ordering
    neighborhood_orderings.append(neighborhood_functions.copy())
    
    # Try some random orderings
    for _ in range(min(5, len(neighborhood_functions))):
        ordering = neighborhood_functions.copy()
        random.shuffle(ordering)
        neighborhood_orderings.append(ordering)
    
    # Run VND with each ordering
    for ordering in neighborhood_orderings:
        # Check time limit
        if time.time() - start_time >= time_limit:
            break
            
        # Determine remaining time
        remaining_time = max(0.1, time_limit - (time.time() - start_time))
        
        # Run VND with this ordering
        solution = vnd_algorithm(
            best_solution,
            release_times,
            processing_times,
            waiting_times,
            penalties,
            ordering,
            max_iterations=max_iterations // len(neighborhood_orderings),
            time_limit=remaining_time
        )
        
        # Evaluate the solution
        solution_value = calculate_solution_value(
            solution, release_times, processing_times, waiting_times, penalties
        )
        
        # Update best solution if needed
        if solution_value < best_value:
            best_solution = copy.deepcopy(solution)
            best_value = solution_value
    
    return best_solution
