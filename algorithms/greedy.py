import numpy as np
from typing import List, Tuple
import random

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

def earliest_ready_time(release_times: List[int], processing_times: List[int], 
                        waiting_times: List[List[int]], penalties: List[int], num_runways: int) -> List[List[int]]:
    """
    Greedy algorithm that schedules flights based on earliest ready time.
    
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
    
    # Create a list of flight indices sorted by release time
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: release_times[i])
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Assign each flight to the runway that can process it the earliest
    for flight in flight_indices:
        min_start_time = float('inf')
        best_runway = -1
        
        # Find the runway that can start this flight the earliest
        for runway in range(num_runways):
            if last_flight_on_runway[runway] == -1:
                # No previous flight on this runway
                possible_start_time = max(runway_times[runway], release_times[flight])
            else:
                # Consider waiting time from previous flight
                prev_flight = last_flight_on_runway[runway]
                possible_start_time = max(
                    runway_times[runway] + waiting_times[prev_flight][flight],
                    release_times[flight]
                )
            
            if possible_start_time < min_start_time:
                min_start_time = possible_start_time
                best_runway = runway
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        runway_times[best_runway] = min_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    return solution

def least_penalty(release_times: List[int], processing_times: List[int], 
                  waiting_times: List[List[int]], penalties: List[int], 
                  num_runways: int) -> List[List[int]]:
    """
    Greedy algorithm that schedules flights to minimize penalties.
    
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
    
    # Create a list of flight indices sorted by penalty value (descending)
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: penalties[i], reverse=True)
    
    # Initialize the solution with empty runways
    solution = [[] for _ in range(num_runways)]
    
    # Initialize the current time for each runway
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways  # No previous flight
    
    # Assign each flight to the runway that minimizes its penalty
    for flight in flight_indices:
        min_penalty = float('inf')
        best_runway = -1
        
        # Find the runway that minimizes the penalty for this flight
        for runway in range(num_runways):
            if last_flight_on_runway[runway] == -1:
                # No previous flight on this runway
                possible_start_time = max(runway_times[runway], release_times[flight])
            else:
                # Consider waiting time from previous flight
                prev_flight = last_flight_on_runway[runway]
                possible_start_time = max(
                    runway_times[runway] + waiting_times[prev_flight][flight],
                    release_times[flight]
                )
            
            # Calculate the delay and penalty
            delay = max(0, possible_start_time - release_times[flight])
            penalty = delay * penalties[flight]
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_runway = runway
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        
        # Update runway time
        if last_flight_on_runway[best_runway] == -1:
            runway_times[best_runway] = max(runway_times[best_runway], release_times[flight]) + processing_times[flight]
        else:
            prev_flight = last_flight_on_runway[best_runway]
            runway_times[best_runway] = max(
                runway_times[best_runway] + waiting_times[prev_flight][flight],
                release_times[flight]
            ) + processing_times[flight]
            
        last_flight_on_runway[best_runway] = flight
    
    return solution

def combined_heuristic(release_times: List[int], processing_times: List[int], 
                       waiting_times: List[List[int]], penalties: List[int], 
                       num_runways: int) -> List[List[int]]:
    """
    Optimized greedy algorithm that combines consideration of release times and penalties.
    
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
    
    # Precompute normalized values once
    max_release = max(release_times) if max(release_times) > 0 else 1
    max_penalty = max(penalties) if max(penalties) > 0 else 1
    
    normalized_release = [rt / max_release for rt in release_times]
    normalized_penalties = [p / max_penalty for p in penalties]
    
    # Precompute combined scores for each flight
    # Lower score = higher priority (early release, high penalty)
    combined_scores = [normalized_release[i] - normalized_penalties[i] for i in range(num_flights)]
    
    # Create a list of flight indices sorted by the combined score
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: combined_scores[i])
    
    # Initialize solution and tracking variables
    solution = [[] for _ in range(num_runways)]
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways
    
    # Precompute waiting time lookups for faster access
    waiting_time_lookup = {}
    for i in range(num_flights):
        for j in range(num_flights):
            waiting_time_lookup[(i, j)] = waiting_times[i][j]
    
    # Assign each flight to minimize delay * penalty
    for flight in flight_indices:
        min_cost = float('inf')
        best_runway = 0
        best_start_time = 0
        
        # Find the runway that minimizes cost
        for runway in range(num_runways):
            # Calculate possible start time on this runway
            if last_flight_on_runway[runway] == -1:
                possible_start_time = max(runway_times[runway], release_times[flight])
            else:
                prev_flight = last_flight_on_runway[runway]
                wait_time = waiting_time_lookup.get((prev_flight, flight), 
                                               waiting_times[prev_flight][flight])
                possible_start_time = max(
                    runway_times[runway] + wait_time,
                    release_times[flight]
                )
            
            # Calculate delay and cost
            delay = max(0, possible_start_time - release_times[flight])
            cost = delay * penalties[flight]
            
            if cost < min_cost:
                min_cost = cost
                best_runway = runway
                best_start_time = possible_start_time
        
        # Assign the flight to the best runway
        solution[best_runway].append(flight)
        
        # Update runway time directly
        runway_times[best_runway] = best_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    return solution

def grasp_construction(release_times: List[int], processing_times: List[int],
                       waiting_times: List[List[int]], penalties: List[int],
                       num_runways: int, alpha: float = 0.3) -> List[List[int]]:
    """
    Constructs a solution using a GRASP-like approach.

    Args:
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        num_runways: Number of available runways
        alpha: GRASP parameter for RCL construction (0 <= alpha <= 1)
               0 = Purely greedy, 1 = Fully random among evaluated

    Returns:
        Assignment of flights to runways
    """
    num_flights = len(release_times)
    unscheduled_flights = set(range(num_flights))

    solution = [[] for _ in range(num_runways)]
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways

    while unscheduled_flights:
        candidate_assignments = []
        min_cost = float('inf')

        # Evaluate all possible next assignments
        for flight in unscheduled_flights:
            for runway in range(num_runways):
                # Calculate possible start time
                if last_flight_on_runway[runway] == -1:
                    possible_start_time = max(runway_times[runway], release_times[flight])
                else:
                    prev_flight = last_flight_on_runway[runway]
                    wait_time = waiting_times[prev_flight][flight]
                    possible_start_time = max(
                        runway_times[runway] + wait_time,
                        release_times[flight]
                    )
                
                # Calculate cost (individual penalty)
                delay = max(0, possible_start_time - release_times[flight])
                cost = delay * penalties[flight]
                
                candidate_assignments.append({
                    'flight': flight,
                    'runway': runway,
                    'start_time': possible_start_time,
                    'cost': cost
                })
                
                min_cost = min(min_cost, cost)
        
        # Build Restricted Candidate List (RCL)
        cost_threshold = min_cost * (1.0 + alpha)
        # Handle case where min_cost is 0: threshold remains 0 unless alpha is large
        if min_cost == 0 and alpha < 1:
            cost_threshold = 0 # Only allow 0-cost moves if possible
        elif min_cost == 0 and alpha >=1:
             # Allow any move if min_cost is 0 and alpha is high (fully random)
             # A small epsilon might be better in practice to avoid truly bad moves
             cost_threshold = float('inf') 
             
        rcl = [cand for cand in candidate_assignments if cand['cost'] <= cost_threshold]

        if not rcl:
            # Fallback if RCL is empty (shouldn't happen if candidates exist)
            rcl = sorted(candidate_assignments, key=lambda x: x['cost'])
            if not rcl: break # No more assignments possible

        # Select randomly from RCL
        selected_assignment = random.choice(rcl)
        selected_flight = selected_assignment['flight']
        selected_runway = selected_assignment['runway']
        selected_start_time = selected_assignment['start_time']

        # Assign the flight
        solution[selected_runway].append(selected_flight)
        runway_times[selected_runway] = selected_start_time + processing_times[selected_flight]
        last_flight_on_runway[selected_runway] = selected_flight
        unscheduled_flights.remove(selected_flight)

    return solution
