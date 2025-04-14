from typing import List, Dict, Any

def calculate_solution_value(solution: List[List[int]], 
                            release_times: List[int], 
                            penalties: List[int]) -> float:
    """
    Calculate the total penalty value of a solution.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Total penalty value
    """
    # In a real implementation, we'd calculate this properly
    # For now, this is a simplified version that doesn't account for waiting times
    
    total_penalty = 0
    
    for runway in solution:
        current_time = 0
        
        for flight in runway:
            # Calculate start time (simplified)
            start_time = max(current_time, release_times[flight])
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            total_penalty += flight_penalty
            
            # Update current time (simplified)
            current_time = start_time + 1  # Just a placeholder
    
    return total_penalty

def calculate_flight_times(solution: List[List[int]], 
                          release_times: List[int], 
                          processing_times: List[int], 
                          waiting_times: List[List[int]]) -> Dict[int, Dict[str, int]]:
    """
    Calculate the start and completion times for all flights in the solution.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        
    Returns:
        Dictionary mapping flight indices to dictionaries with start and end times
    """
    flight_times = {}
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate end time
            end_time = start_time + processing_times[flight]
            
            # Store times
            flight_times[flight] = {
                'start_time': start_time,
                'end_time': end_time,
                'runway': runway_idx
            }
            
            # Update current time and previous flight
            current_time = end_time
            prev_flight = flight
    
    return flight_times

def calculate_delays(solution: List[List[int]], 
                    release_times: List[int], 
                    processing_times: List[int], 
                    waiting_times: List[List[int]]) -> Dict[int, int]:
    """
    Calculate the delay for each flight in the solution.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        
    Returns:
        Dictionary mapping flight indices to delay values
    """
    delays = {}
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate delay
            delay = max(0, start_time - release_times[flight])
            delays[flight] = delay
            
            # Update current time and previous flight
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    return delays

def calculate_gap(solution_value: float, optimal_value: float) -> float:
    """
    Calculate the GAP between a solution and the optimal value.
    
    Args:
        solution_value: Value of the solution
        optimal_value: Optimal solution value
        
    Returns:
        GAP as a percentage
    """
    if optimal_value == 0:
        return 0 if solution_value == 0 else float('inf')
    
    gap = ((solution_value - optimal_value) / optimal_value) * 100
    return gap

def get_solution_statistics(solution: List[List[int]], 
                           release_times: List[int], 
                           processing_times: List[int], 
                           waiting_times: List[List[int]], 
                           penalties: List[int]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a solution.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        waiting_times: Matrix of required waiting times between consecutive flights
        penalties: Penalty per unit time for each flight
        
    Returns:
        Dictionary of statistics
    """
    # Calculate flight times
    flight_times = calculate_flight_times(solution, release_times, processing_times, waiting_times)
    
    # Calculate delays
    delays = calculate_delays(solution, release_times, processing_times, waiting_times)
    
    # Calculate penalties
    flight_penalties = {}
    total_penalty = 0
    for flight, delay in delays.items():
        penalty = delay * penalties[flight]
        flight_penalties[flight] = penalty
        total_penalty += penalty
    
    # Calculate runway utilization
    runway_utilization = {}
    for runway_idx in range(len(solution)):
        runway = solution[runway_idx]
        if not runway:
            runway_utilization[runway_idx] = 0
            continue
        
        # Calculate total processing time on this runway
        total_time = 0
        for flight in runway:
            total_time += processing_times[flight]
        
        # Calculate span of the runway (from first start to last end)
        start_times = [flight_times[flight]['start_time'] for flight in runway]
        end_times = [flight_times[flight]['end_time'] for flight in runway]
        span = max(end_times) - min(start_times)
        
        # Calculate utilization as percentage
        utilization = (total_time / span) * 100 if span > 0 else 0
        runway_utilization[runway_idx] = utilization
    
    # Compile statistics
    statistics = {
        'total_penalty': total_penalty,
        'average_delay': sum(delays.values()) / len(delays) if delays else 0,
        'max_delay': max(delays.values()) if delays else 0,
        'average_penalty': sum(flight_penalties.values()) / len(flight_penalties) if flight_penalties else 0,
        'max_penalty': max(flight_penalties.values()) if flight_penalties else 0,
        'runway_utilization': runway_utilization,
        'average_utilization': sum(runway_utilization.values()) / len(runway_utilization) if runway_utilization else 0,
        'flights_per_runway': [len(runway) for runway in solution],
        'flight_times': flight_times,
        'delays': delays,
        'flight_penalties': flight_penalties
    }
    
    return statistics
