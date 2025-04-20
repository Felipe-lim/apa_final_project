import random
import copy
from typing import List, Tuple, Set

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

def reinsert_delayed_flight(solution: List[List[int]], 
                           release_times: List[int], 
                           processing_times: List[int], 
                           waiting_times: List[List[int]], 
                           penalties: List[int]) -> List[List[int]]:
    """
    Find a delayed flight and try to reinsert it in a better position.
    Prioritizes flights with high delay * penalty.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with a delayed flight repositioned
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # If there are no runways or all are empty, return unchanged
    if not solution or all(len(runway) == 0 for runway in solution):
        return new_solution
    
    # Calculate delays and penalties for each flight
    flight_delays = {}
    flight_penalties = {}
    flight_runways = {}
    flight_positions = {}
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for pos, flight in enumerate(runway):
            # Calculate start time
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_delays[flight] = delay
            flight_penalties[flight] = delay * penalties[flight]
            flight_runways[flight] = runway_idx
            flight_positions[flight] = pos
            
            # Update current time and previous flight
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    # Find flights with delays
    delayed_flights = [(f, p) for f, p in flight_penalties.items() if p > 0]
    
    if not delayed_flights:
        return new_solution  # No delayed flights, return unchanged
    
    # Sort by penalty value (descending)
    delayed_flights.sort(key=lambda x: x[1], reverse=True)
    
    # Try to reposition the most penalized flights
    for flight, _ in delayed_flights[:min(5, len(delayed_flights))]:
        # Get current position
        if flight not in flight_runways or flight not in flight_positions:
            continue
            
        current_runway = flight_runways[flight]
        current_pos = flight_positions[flight]
        
        # Validate current_runway and current_pos are valid
        if (current_runway >= len(new_solution) or 
            current_pos >= len(new_solution[current_runway])):
            continue
        
        # Remove flight from current runway
        flight_obj = new_solution[current_runway].pop(current_pos)
        
        best_penalty = float('inf')
        best_runway = current_runway
        best_pos = current_pos
        
        # Try inserting into each runway at each position
        for runway_idx in range(len(new_solution)):
            runway = new_solution[runway_idx]
            
            for insert_pos in range(len(runway) + 1):
                # Insert flight temporarily
                runway.insert(insert_pos, flight_obj)
                
                # Calculate penalty for this runway only
                penalty = 0
                current_time = 0
                prev_f = None
                
                for f in runway:
                    if prev_f is None:
                        start_time = max(current_time, release_times[f])
                    else:
                        min_start_time = current_time + waiting_times[prev_f][f]
                        start_time = max(min_start_time, release_times[f])
                    
                    delay = max(0, start_time - release_times[f])
                    penalty += delay * penalties[f]
                    
                    current_time = start_time + processing_times[f]
                    prev_f = f
                
                # Remove flight again
                runway.pop(insert_pos)
                
                # Check if this position is better
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_runway = runway_idx
                    best_pos = insert_pos
        
        # Insert at best position found
        new_solution[best_runway].insert(best_pos, flight_obj)
    
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
        reinsert_flight,
        reinsert_delayed_flight
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

def optimize_runway_balance(solution: List[List[int]], 
                           release_times: List[int], 
                           processing_times: List[int], 
                           waiting_times: List[List[int]], 
                           penalties: List[int]) -> List[List[int]]:
    """
    Try to balance the number of flights between runways to reduce congestion.
    
    Args:
        solution: Current assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Modified solution with more balanced runways
    """
    # Create a deep copy of the solution to avoid modifying the original
    new_solution = copy.deepcopy(solution)
    
    # Calculate the average number of flights per runway
    total_flights = sum(len(runway) for runway in new_solution)
    avg_flights = total_flights / len(new_solution)
    
    # Find the most overloaded and underloaded runways
    runway_loads = [(i, len(runway)) for i, runway in enumerate(new_solution)]
    runway_loads.sort(key=lambda x: x[1], reverse=True)
    
    # If runways are already balanced, return unchanged
    if len(runway_loads) <= 1 or runway_loads[0][1] - runway_loads[-1][1] <= 1:
        return new_solution
    
    # Try to move a flight from the most overloaded to the most underloaded runway
    overloaded_idx = runway_loads[0][0]
    underloaded_idx = runway_loads[-1][0]
    
    overloaded_runway = new_solution[overloaded_idx]
    underloaded_runway = new_solution[underloaded_idx]
    
    # Find the best flight to move (least penalty increase)
    best_penalty_increase = float('inf')
    best_flight_pos = -1
    best_insert_pos = -1
    
    # Calculate current penalty for both runways
    current_penalty = 0
    
    # Overloaded runway
    current_time = 0
    prev_flight = None
    for flight in overloaded_runway:
        if prev_flight is None:
            start_time = max(current_time, release_times[flight])
        else:
            min_start_time = current_time + waiting_times[prev_flight][flight]
            start_time = max(min_start_time, release_times[flight])
        
        delay = max(0, start_time - release_times[flight])
        current_penalty += delay * penalties[flight]
        
        current_time = start_time + processing_times[flight]
        prev_flight = flight
    
    # Underloaded runway
    current_time = 0
    prev_flight = None
    for flight in underloaded_runway:
        if prev_flight is None:
            start_time = max(current_time, release_times[flight])
        else:
            min_start_time = current_time + waiting_times[prev_flight][flight]
            start_time = max(min_start_time, release_times[flight])
        
        delay = max(0, start_time - release_times[flight])
        current_penalty += delay * penalties[flight]
        
        current_time = start_time + processing_times[flight]
        prev_flight = flight
    
    # Try each flight in the overloaded runway
    for flight_pos, flight in enumerate(overloaded_runway):
        for insert_pos in range(len(underloaded_runway) + 1):
            # Temporarily move the flight
            test_overloaded = overloaded_runway.copy()
            test_underloaded = underloaded_runway.copy()
            
            flight_obj = test_overloaded.pop(flight_pos)
            test_underloaded.insert(insert_pos, flight_obj)
            
            # Calculate new penalty
            new_penalty = 0
            
            # Modified overloaded runway
            current_time = 0
            prev_flight = None
            for f in test_overloaded:
                if prev_flight is None:
                    start_time = max(current_time, release_times[f])
                else:
                    min_start_time = current_time + waiting_times[prev_flight][f]
                    start_time = max(min_start_time, release_times[f])
                
                delay = max(0, start_time - release_times[f])
                new_penalty += delay * penalties[f]
                
                current_time = start_time + processing_times[f]
                prev_flight = f
            
            # Modified underloaded runway
            current_time = 0
            prev_flight = None
            for f in test_underloaded:
                if prev_flight is None:
                    start_time = max(current_time, release_times[f])
                else:
                    min_start_time = current_time + waiting_times[prev_flight][f]
                    start_time = max(min_start_time, release_times[f])
                
                delay = max(0, start_time - release_times[f])
                new_penalty += delay * penalties[f]
                
                current_time = start_time + processing_times[f]
                prev_flight = f
            
            # Calculate penalty change
            penalty_increase = new_penalty - current_penalty
            
            if penalty_increase < best_penalty_increase:
                best_penalty_increase = penalty_increase
                best_flight_pos = flight_pos
                best_insert_pos = insert_pos
    
    # If we found a good move, apply it
    if best_flight_pos != -1:
        flight_obj = new_solution[overloaded_idx].pop(best_flight_pos)
        new_solution[underloaded_idx].insert(best_insert_pos, flight_obj)
    
    return new_solution
