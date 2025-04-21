from typing import List, Tuple
import random

def combined_heuristic(release_times: List[int], processing_times: List[int], 
                       waiting_times: List[List[int]], penalties: List[int], 
                       num_runways: int) -> List[List[int]]:
    num_flights = len(release_times)
    
    max_release = max(release_times) if max(release_times) > 0 else 1
    max_penalty = max(penalties) if max(penalties) > 0 else 1
    
    normalized_release = [rt / max_release for rt in release_times]
    normalized_penalties = [p / max_penalty for p in penalties]
    
    combined_scores = [normalized_release[i] - normalized_penalties[i] for i in range(num_flights)]
    
    flight_indices = list(range(num_flights))
    flight_indices.sort(key=lambda i: combined_scores[i])
    
    solution = [[] for _ in range(num_runways)]
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways
    
    waiting_time_lookup = {}
    for i in range(num_flights):
        for j in range(num_flights):
            waiting_time_lookup[(i, j)] = waiting_times[i][j]
    
    for flight in flight_indices:
        min_cost = float('inf')
        best_runway = 0
        best_start_time = 0
        
        for runway in range(num_runways):
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
            
            delay = max(0, possible_start_time - release_times[flight])
            cost = delay * penalties[flight]
            
            if cost < min_cost:
                min_cost = cost
                best_runway = runway
                best_start_time = possible_start_time
        
        solution[best_runway].append(flight)
        
        runway_times[best_runway] = best_start_time + processing_times[flight]
        last_flight_on_runway[best_runway] = flight
    
    return solution

def grasp_construction(release_times: List[int], processing_times: List[int],
                       waiting_times: List[List[int]], penalties: List[int],
                       num_runways: int, alpha: float = 0.3) -> List[List[int]]:
    num_flights = len(release_times)
    unscheduled_flights = set(range(num_flights))

    solution = [[] for _ in range(num_runways)]
    runway_times = [0] * num_runways
    last_flight_on_runway = [-1] * num_runways

    while unscheduled_flights:
        candidate_assignments = []
        min_cost = float('inf')

        for flight in unscheduled_flights:
            for runway in range(num_runways):
                if last_flight_on_runway[runway] == -1:
                    possible_start_time = max(runway_times[runway], release_times[flight])
                else:
                    prev_flight = last_flight_on_runway[runway]
                    wait_time = waiting_times[prev_flight][flight]
                    possible_start_time = max(
                        runway_times[runway] + wait_time,
                        release_times[flight]
                    )
                
                delay = max(0, possible_start_time - release_times[flight])
                cost = delay * penalties[flight]
                
                candidate_assignments.append({
                    'flight': flight,
                    'runway': runway,
                    'start_time': possible_start_time,
                    'cost': cost
                })
                
                min_cost = min(min_cost, cost)
        
        cost_threshold = min_cost * (1.0 + alpha)
        if min_cost == 0 and alpha < 1:
            cost_threshold = 0
        elif min_cost == 0 and alpha >=1:
             cost_threshold = float('inf') 
             
        rcl = [cand for cand in candidate_assignments if cand['cost'] <= cost_threshold]

        if not rcl:
            rcl = sorted(candidate_assignments, key=lambda x: x['cost'])
            if not rcl: break

        selected_assignment = random.choice(rcl)
        selected_flight = selected_assignment['flight']
        selected_runway = selected_assignment['runway']
        selected_start_time = selected_assignment['start_time']

        solution[selected_runway].append(selected_flight)
        runway_times[selected_runway] = selected_start_time + processing_times[selected_flight]
        last_flight_on_runway[selected_runway] = selected_flight
        unscheduled_flights.remove(selected_flight)

    return solution