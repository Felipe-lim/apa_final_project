import numpy as np
import random
import copy
from typing import List, Tuple, Dict, Any, Optional

def calculate_completion_times(solution: List[List[int]], release_times: List[int], 
                               processing_times: List[int], waiting_times: List[List[int]]) -> List[int]:
    """
    Calculate the completion times for all flights in the current solution.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        
    Returns:
        List of completion times for each flight
    """
    num_flights = len(release_times)
    completion_times = [-1] * num_flights  # -1 means unscheduled
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time for this flight
            if prev_flight is None:
                # First flight on this runway
                start_time = max(current_time, release_times[flight])
            else:
                # Consider waiting time from previous flight
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate completion time
            completion_time = start_time + processing_times[flight]
            completion_times[flight] = completion_time
            
            # Update current time and previous flight
            current_time = completion_time
            prev_flight = flight
    
    return completion_times

def calculate_penalties(solution: List[List[int]], release_times: List[int], 
                        processing_times: List[int], waiting_times: List[List[int]], 
                        penalties: List[int]) -> List[float]:
    """
    Calculate penalties for all flights in the given solution.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        List of penalties for each flight
    """
    num_flights = len(release_times)
    flight_penalties = [0] * num_flights
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time for this flight
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_penalties[flight] = delay * penalties[flight]
            
            # Update current time and previous flight
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    return flight_penalties

def simulate_runway_assignment(flight: int, 
                              runway_idx: int, 
                              solution: List[List[int]],
                              runway_times: List[int],
                              last_flight_on_runway: List[int], 
                              release_times: List[int],
                              processing_times: List[int],
                              waiting_times: List[List[int]]) -> Tuple[int, int]:
    """
    Simulate assigning a flight to a runway and return the start time and completion time.
    
    Args:
        flight: Flight index to assign
        runway_idx: Runway index to assign the flight to
        solution: Current assignment of flights to runways
        runway_times: Current time for each runway
        last_flight_on_runway: Last flight assigned to each runway
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        
    Returns:
        Tuple containing start time and completion time
    """
    if last_flight_on_runway[runway_idx] == -1:
        # No previous flight on this runway
        start_time = max(runway_times[runway_idx], release_times[flight])
    else:
        # Consider waiting time from previous flight
        prev_flight = last_flight_on_runway[runway_idx]
        start_time = max(
            runway_times[runway_idx] + waiting_times[prev_flight][flight],
            release_times[flight]
        )
    
    # Calculate completion time
    completion_time = start_time + processing_times[flight]
    
    return start_time, completion_time

def calculate_total_penalty(solution: List[List[int]], 
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

def earliest_ready_time(release_times: List[int], processing_times: List[int], 
                        waiting_times: List[List[int]], num_runways: int) -> List[List[int]]:
    """
    Improved greedy algorithm that schedules flights based on earliest ready time.
    
    Args:
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        num_runways: Number of available runways
        
    Returns:
        Assignment of flights to runways
    """
    num_flights = len(release_times)
    
    # Create a default penalty list (all 1's) for sorting
    default_penalties = [1] * num_flights
    
    # Create a list of flight indices sorted by release time
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: (release_times[i], -default_penalties[i]))
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Assign each flight to the runway that can process it the earliest
    for flight in flight_indices:
        min_start_time = float('inf')
        best_runway = -1
        
        # Try each possible position in each runway to find the best position
        for runway in range(num_runways):
            # Simulate adding the flight to the end of this runway
            start_time, completion_time = simulate_runway_assignment(
                flight, runway, solution, runway_times, last_flight_on_runway,
                release_times, processing_times, waiting_times
            )
            
            if start_time < min_start_time:
                min_start_time = start_time
                best_runway = runway
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        runway_times[best_runway] = min_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    # Add a local search phase to improve the initial solution
    # Use default penalties of all 1's if not provided
    improved_solution = improve_solution_with_local_search(
        solution, release_times, processing_times, waiting_times, 
        default_penalties,
        max_iterations=100
    )
    
    return improved_solution

def least_penalty(release_times: List[int], processing_times: List[int], 
                  waiting_times: List[List[int]], penalties: List[int], 
                  num_runways: int) -> List[List[int]]:
    """
    Improved greedy algorithm that schedules flights to minimize penalties.
    
    Args:
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        num_runways: Number of available runways
        
    Returns:
        Assignment of flights to runways
    """
    num_flights = len(release_times)
    
    # Create a list of flight indices sorted by penalty value (descending) and release time (ascending)
    # This improves the balance between high penalty flights and early release times
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: (-penalties[i], release_times[i]))
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Assign each flight to the runway that minimizes its penalty
    for flight in flight_indices:
        min_penalty = float('inf')
        best_runway = -1
        best_start_time = 0
        
        # Evaluate each runway to find the one that minimizes penalty
        for runway in range(num_runways):
            # Simulate adding the flight to this runway
            start_time, completion_time = simulate_runway_assignment(
                flight, runway, solution, runway_times, last_flight_on_runway,
                release_times, processing_times, waiting_times
            )
            
            # Calculate the delay and penalty
            delay = max(0, start_time - release_times[flight])
            penalty = delay * penalties[flight]
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_runway = runway
                best_start_time = start_time
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        
        # Update runway time correctly
        runway_times[best_runway] = best_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    # Apply local search to improve the initial solution
    improved_solution = improve_solution_with_local_search(
        solution, release_times, processing_times, waiting_times, penalties,
        max_iterations=100
    )
    
    return improved_solution

def combined_heuristic(release_times: List[int], processing_times: List[int], 
                       waiting_times: List[List[int]], penalties: List[int], 
                       num_runways: int) -> List[List[int]]:
    """
    Enhanced greedy algorithm that combines consideration of release times and penalties
    with look-ahead capability.
    
    Args:
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        num_runways: Number of available runways
        
    Returns:
        Assignment of flights to runways
    """
    num_flights = len(release_times)
    
    # Normalize values for better comparison
    max_release = max(release_times) if max(release_times) > 0 else 1
    max_penalty = max(penalties) if max(penalties) > 0 else 1
    
    # Calculate a priority score for each flight - weighted combination of factors
    # Consider both release time (earlier = better) and penalty (higher = better)
    weight_release = 0.4
    weight_penalty = 0.6
    
    flight_scores = []
    for i in range(num_flights):
        # Normalize the values between 0 and 1
        norm_release = 1.0 - (release_times[i] / max_release)  # Invert so earlier is better
        norm_penalty = penalties[i] / max_penalty
        
        # Combined score - higher is better priority
        score = (weight_release * norm_release) + (weight_penalty * norm_penalty)
        flight_scores.append((i, score))
    
    # Sort flights by score (descending)
    flight_scores.sort(key=lambda x: x[1], reverse=True)
    flight_indices = [i for i, _ in flight_scores]
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Assign flights using a more sophisticated approach with look-ahead
    for flight_idx, flight in enumerate(flight_indices):
        min_cost = float('inf')
        best_runway = -1
        best_start_time = 0
        
        # For each runway, calculate the immediate cost of assigning this flight
        # plus an estimate of the impact on future flights
        for runway in range(num_runways):
            # Calculate immediate impact
            start_time, completion_time = simulate_runway_assignment(
                flight, runway, solution, runway_times, last_flight_on_runway,
                release_times, processing_times, waiting_times
            )
            
            # Calculate immediate delay and cost
            delay = max(0, start_time - release_times[flight])
            immediate_cost = delay * penalties[flight]
            
            # Estimate impact on future flights (look-ahead)
            future_impact = 0
            if flight_idx < len(flight_indices) - 1:
                # Look at the next few flights (up to 3) that would be affected
                look_ahead_count = min(3, len(flight_indices) - flight_idx - 1)
                for la_idx in range(1, look_ahead_count + 1):
                    next_flight = flight_indices[flight_idx + la_idx]
                    
                    # Estimate the impact if this next flight were to follow on this runway
                    future_start_time = max(
                        completion_time + waiting_times[flight][next_flight],
                        release_times[next_flight]
                    )
                    future_delay = max(0, future_start_time - release_times[next_flight])
                    future_impact += (future_delay * penalties[next_flight]) / la_idx  # Discount by distance in the future
            
            # Total cost is immediate plus a fraction of estimated future impact
            total_cost = immediate_cost + (0.5 * future_impact)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_runway = runway
                best_start_time = start_time
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        
        # Update runway time
        runway_times[best_runway] = best_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    # Apply local search to improve the initial solution
    improved_solution = improve_solution_with_local_search(
        solution, release_times, processing_times, waiting_times, penalties,
        max_iterations=150
    )
    
    return improved_solution

def regret_heuristic(release_times: List[int], processing_times: List[int], 
                     waiting_times: List[List[int]], penalties: List[int], 
                     num_runways: int) -> List[List[int]]:
    """
    Advanced regret-based construction heuristic that considers
    the difference in penalties between assigning a flight to its best runway
    versus its second-best runway.
    
    Args:
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        num_runways: Number of available runways
        
    Returns:
        Assignment of flights to runways
    """
    num_flights = len(release_times)
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Keep track of flights that haven't been assigned yet
    unassigned_flights = set(range(num_flights))
    
    # Main loop - assign flights one by one based on regret value
    while unassigned_flights:
        max_regret = -1
        best_flight = -1
        best_runway = -1
        best_start_time = 0
        
        # For each unassigned flight, calculate its regret value
        for flight in unassigned_flights:
            # Calculate the cost for each runway
            runway_costs = []
            runway_start_times = []
            
            for runway in range(num_runways):
                # Simulate adding this flight to the runway
                start_time, completion_time = simulate_runway_assignment(
                    flight, runway, solution, runway_times, last_flight_on_runway,
                    release_times, processing_times, waiting_times
                )
                
                # Calculate the delay and cost
                delay = max(0, start_time - release_times[flight])
                cost = delay * penalties[flight]
                
                runway_costs.append(cost)
                runway_start_times.append(start_time)
            
            # Sort costs to find best and second-best
            sorted_costs = sorted(runway_costs)
            
            # Calculate regret (difference between best and second-best)
            if len(sorted_costs) >= 2:
                regret = sorted_costs[1] - sorted_costs[0]
            else:
                regret = sorted_costs[0]
            
            # Consider both regret and absolute cost
            # If regret is high but cost is very high as well, we might want to prioritize lower cost
            adjusted_regret = regret * (1 + 1 / (sorted_costs[0] + 1))  # +1 to avoid division by zero
            
            if adjusted_regret > max_regret:
                max_regret = adjusted_regret
                best_flight = flight
                best_runway = runway_costs.index(sorted_costs[0])  # Get the index of the best runway
                best_start_time = runway_start_times[best_runway]
        
        # Assign the flight with the highest regret to its best runway
        solution[best_runway].append(best_flight)
        unassigned_flights.remove(best_flight)
        
        # Update runway time
        runway_times[best_runway] = best_start_time + processing_times[best_flight]
        last_flight_on_runway[best_runway] = best_flight
    
    # Apply local search to improve the initial solution
    improved_solution = improve_solution_with_local_search(
        solution, release_times, processing_times, waiting_times, penalties,
        max_iterations=200  # More iterations for this advanced heuristic
    )
    
    return improved_solution

def improve_solution_with_local_search(solution: List[List[int]],
                                     release_times: List[int],
                                     processing_times: List[int],
                                     waiting_times: List[List[int]],
                                     penalties: List[int],
                                     max_iterations: int = 100) -> List[List[int]]:
    """
    Apply a simple local search to improve the solution.
    
    Args:
        solution: Initial assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        max_iterations: Maximum number of local search iterations
        
    Returns:
        Improved solution
    """
    # Make a deep copy of the solution to avoid modifying the original
    current_solution = copy.deepcopy(solution)
    current_value = calculate_total_penalty(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    iteration = 0
    while iteration < max_iterations:
        improved = False
        
        # Try to move a flight to a different runway
        for source_runway in range(len(current_solution)):
            if not current_solution[source_runway]:
                continue
                
            for flight_idx, flight in enumerate(current_solution[source_runway]):
                for target_runway in range(len(current_solution)):
                    if target_runway == source_runway:
                        continue
                    
                    # Create a new solution by moving the flight
                    new_solution = copy.deepcopy(current_solution)
                    moved_flight = new_solution[source_runway].pop(flight_idx)
                    
                    # Try inserting at different positions in the target runway
                    best_insert_pos = len(new_solution[target_runway])
                    best_insert_value = float('inf')
                    
                    for insert_pos in range(len(new_solution[target_runway]) + 1):
                        # Insert the flight
                        test_solution = copy.deepcopy(new_solution)
                        test_solution[target_runway].insert(insert_pos, moved_flight)
                        
                        # Evaluate the solution
                        test_value = calculate_total_penalty(
                            test_solution, release_times, processing_times, waiting_times, penalties
                        )
                        
                        if test_value < best_insert_value:
                            best_insert_value = test_value
                            best_insert_pos = insert_pos
                    
                    # Insert at the best position
                    new_solution[target_runway].insert(best_insert_pos, moved_flight)
                    
                    # Calculate the value of the new solution
                    new_value = calculate_total_penalty(
                        new_solution, release_times, processing_times, waiting_times, penalties
                    )
                    
                    # If it's better, update the current solution
                    if new_value < current_value:
                        current_solution = new_solution
                        current_value = new_value
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        # If no improvement was found, try swapping flights between runways
        if not improved:
            for runway1 in range(len(current_solution)):
                if not current_solution[runway1]:
                    continue
                    
                for idx1, flight1 in enumerate(current_solution[runway1]):
                    for runway2 in range(runway1 + 1, len(current_solution)):
                        if not current_solution[runway2]:
                            continue
                            
                        for idx2, flight2 in enumerate(current_solution[runway2]):
                            # Create a new solution by swapping the flights
                            new_solution = copy.deepcopy(current_solution)
                            new_solution[runway1][idx1] = flight2
                            new_solution[runway2][idx2] = flight1
                            
                            # Calculate the value of the new solution
                            new_value = calculate_total_penalty(
                                new_solution, release_times, processing_times, waiting_times, penalties
                            )
                            
                            # If it's better, update the current solution
                            if new_value < current_value:
                                current_solution = new_solution
                                current_value = new_value
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        # If no improvement was found, end the search
        if not improved:
            break
        
        iteration += 1
    
    return current_solution
