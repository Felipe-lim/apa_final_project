import random
import copy
from typing import List, Tuple

def swap_flights(solution: List[List[int]], 
                release_times: List[int], 
                processing_times: List[int], 
                waiting_times: List[List[int]], 
                penalties: List[int]) -> List[List[int]]:
    """
    Swap two consecutive flights on the same runway.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with flights swapped
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # Collect runways with at least 2 flights
    eligible_runways = []
    for i, runway in enumerate(new_solution):
        if len(runway) >= 2:
            eligible_runways.append(i)
    
    if not eligible_runways:
        return new_solution  # No eligible runways, return unchanged
    
    # Select a random runway
    runway_idx = random.choice(eligible_runways)
    runway = new_solution[runway_idx]
    
    # Select a random position for swapping (must have a next flight)
    swap_pos = random.randint(0, len(runway) - 2)
    
    # Swap the flights
    runway[swap_pos], runway[swap_pos + 1] = runway[swap_pos + 1], runway[swap_pos]
    
    return new_solution

def move_flight(solution: List[List[int]], 
               release_times: List[int], 
               processing_times: List[int], 
               waiting_times: List[List[int]], 
               penalties: List[int]) -> List[List[int]]:
    """
    Move a flight to a different position in the same runway.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with a flight moved
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # Collect runways with at least 2 flights
    eligible_runways = []
    for i, runway in enumerate(new_solution):
        if len(runway) >= 2:
            eligible_runways.append(i)
    
    if not eligible_runways:
        return new_solution  # No eligible runways, return unchanged
    
    # Select a random runway
    runway_idx = random.choice(eligible_runways)
    runway = new_solution[runway_idx]
    
    # Select a random flight to move
    from_pos = random.randint(0, len(runway) - 1)
    
    # Select a random position to move it to (different from original)
    possible_positions = list(range(len(runway)))
    possible_positions.remove(from_pos)
    to_pos = random.choice(possible_positions)
    
    # Store the flight to be moved
    flight = runway[from_pos]
    
    # Remove the flight from its original position
    runway.pop(from_pos)
    
    # Insert the flight at the new position
    runway.insert(to_pos, flight)
    
    return new_solution

def swap_flights_between_runways(solution: List[List[int]], 
                                release_times: List[int], 
                                processing_times: List[int], 
                                waiting_times: List[List[int]], 
                                penalties: List[int]) -> List[List[int]]:
    """
    Swap flights between different runways.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with flights swapped between runways
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # We need at least 2 runways with at least 1 flight each
    non_empty_runways = []
    for i, runway in enumerate(new_solution):
        if len(runway) >= 1:
            non_empty_runways.append(i)
    
    if len(non_empty_runways) < 2:
        return new_solution  # Not enough eligible runways, return unchanged
    
    # Select two different runways
    runway1_idx, runway2_idx = random.sample(non_empty_runways, 2)
    
    runway1 = new_solution[runway1_idx]
    runway2 = new_solution[runway2_idx]
    
    # Select a random flight from each runway
    flight1_pos = random.randint(0, len(runway1) - 1)
    flight2_pos = random.randint(0, len(runway2) - 1)
    
    # Swap the flights
    runway1[flight1_pos], runway2[flight2_pos] = runway2[flight2_pos], runway1[flight1_pos]
    
    return new_solution

def reinsert_flight(solution: List[List[int]], 
                   release_times: List[int], 
                   processing_times: List[int], 
                   waiting_times: List[List[int]], 
                   penalties: List[int]) -> List[List[int]]:
    """
    Remove a flight from one runway and insert it into another runway.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with a flight reinserted
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # Find runways with at least one flight
    non_empty_runways = []
    for i, runway in enumerate(new_solution):
        if len(runway) >= 1:
            non_empty_runways.append(i)
    
    if not non_empty_runways:
        return new_solution  # No eligible runways, return unchanged
    
    # Select a random source runway
    source_runway_idx = random.choice(non_empty_runways)
    source_runway = new_solution[source_runway_idx]
    
    # Select a random flight to move
    flight_pos = random.randint(0, len(source_runway) - 1)
    flight = source_runway.pop(flight_pos)
    
    # Select a random target runway (can be the same as source)
    target_runway_idx = random.randint(0, len(new_solution) - 1)
    target_runway = new_solution[target_runway_idx]
    
    # Select a random position to insert
    insert_pos = random.randint(0, len(target_runway))
    target_runway.insert(insert_pos, flight)
    
    return new_solution

def shake_solution(solution: List[List[int]], 
                  release_times: List[int], 
                  processing_times: List[int], 
                  waiting_times: List[List[int]], 
                  penalties: List[int],
                  intensity: int = 3) -> List[List[int]]:
    """
    Perform multiple random moves to shake the solution (useful for diversification).
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        intensity: Number of moves to perform
        
    Returns:
        Shaken solution
    """
    new_solution = copy.deepcopy(solution)
    
    # List of available neighborhood functions
    neighborhoods = [
        swap_flights,
        move_flight,
        swap_flights_between_runways,
        reinsert_flight
    ]
    
    # Perform multiple random moves
    for _ in range(intensity):
        # Select a random neighborhood function
        neighborhood_function = random.choice(neighborhoods)
        
        # Apply the function
        new_solution = neighborhood_function(
            new_solution, release_times, processing_times, waiting_times, penalties
        )
    
    return new_solution
