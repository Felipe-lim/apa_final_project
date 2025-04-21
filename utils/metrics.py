from typing import List, Dict, Any

def calculate_solution_value(solution: List[List[int]], 
                            release_times: List[int], 
                            processing_times: List[int] | None = None,
                            waiting_times: List[List[int]] | None = None,
                            penalties: List[int] | None = None) -> float:

    total_penalty = 0
    
    for runway in solution:
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                if waiting_times is not None:
                    wait_time = waiting_times[prev_flight][flight]
                    earliest_possible_time = current_time + wait_time
                    start_time = max(earliest_possible_time, release_times[flight])
                else:
                    start_time = max(current_time, release_times[flight])
            
            delay = max(0, start_time - release_times[flight])
            if penalties is not None:
                flight_penalty = delay * penalties[flight]
                total_penalty += flight_penalty
            
            if processing_times is not None:
                current_time = start_time + processing_times[flight]
            else:
                current_time = start_time + 1
            prev_flight = flight
    
    return total_penalty

def calculate_gap(solution_value: float, optimal_value: float) -> float:

    if optimal_value == 0:
        return 0 if solution_value == 0 else float('inf')
    
    gap = ((solution_value - optimal_value) / optimal_value) * 100
    return gap
