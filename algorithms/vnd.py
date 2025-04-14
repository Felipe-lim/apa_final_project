from typing import List, Callable, Tuple
import copy
import random

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
                 max_iterations: int = 100) -> List[List[int]]:
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
    
    while k < len(neighborhood_functions) and iteration < max_iterations:
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
            k = 0  # Reset to the first neighborhood
        else:
            # Move to the next neighborhood
            k += 1
        
        iteration += 1
    
    return current_solution

def first_improvement_vnd(initial_solution: List[List[int]],
                         release_times: List[int],
                         processing_times: List[int],
                         waiting_times: List[List[int]],
                         penalties: List[int],
                         neighborhood_functions: List[Callable],
                         max_iterations: int = 100) -> List[List[int]]:
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
    
    while iteration < max_iterations:
        improvement_found = False
        
        # Try each neighborhood in sequence
        for neighborhood_function in neighborhood_functions:
            # Generate multiple neighbors and accept the first improvement
            for _ in range(min(10, max_iterations - iteration)):  # Generate up to 10 neighbors
                neighbor = neighborhood_function(
                    current_solution, release_times, processing_times, waiting_times, penalties
                )
                
                neighbor_value = calculate_solution_value(
                    neighbor, release_times, processing_times, waiting_times, penalties
                )
                
                if neighbor_value < current_value:
                    current_solution = copy.deepcopy(neighbor)
                    current_value = neighbor_value
                    improvement_found = True
                    break  # Found improvement, break inner loop
            
            if improvement_found:
                break  # Found improvement, break outer loop
        
        # If no improvement was found in any neighborhood, stop
        if not improvement_found:
            break
        
        iteration += 1
    
    return current_solution
