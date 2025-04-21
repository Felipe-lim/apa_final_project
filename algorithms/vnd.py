from typing import List, Callable, Tuple
import copy
import random
import time

def calculate_solution_value(solution: List[List[int]], 
                            release_times: List[int], 
                            processing_times: List[int], 
                            waiting_times: List[List[int]], 
                            penalties: List[int]) -> float:
    total_penalty = 0
    
    for runway in solution:
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
            total_penalty += flight_penalty
            
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
    
    return new_penalty - original_penalty

def vnd_algorithm(initial_solution: List[List[int]],
                 release_times: List[int],
                 processing_times: List[int],
                 waiting_times: List[List[int]],
                 penalties: List[int],
                 neighborhood_functions: List[Callable],
                 max_iterations: int = 1000,
                 time_limit: float = 10.0) -> List[List[int]]:
    if not neighborhood_functions:
        return initial_solution
    
    current_solution = copy.deepcopy(initial_solution)
    current_value = calculate_solution_value(
        current_solution, release_times, processing_times, waiting_times, penalties
    )
    
    iteration = 0
    k = 0
    start_time = time.time()
    
    no_improvement_count = 0
    max_no_improvement = 5
    
    best_solution = copy.deepcopy(current_solution)
    best_value = current_value
    
    time_check_interval = 50
    
    while iteration < max_iterations and no_improvement_count < max_no_improvement:
        if iteration % time_check_interval == 0 and time.time() - start_time > time_limit:
            break
            
        neighborhood_function = neighborhood_functions[k]
        
        neighbor = neighborhood_function(
            current_solution, release_times, processing_times, waiting_times, penalties
        )
        
        modified_runways = []
        for i in range(min(len(current_solution), len(neighbor))):
            if current_solution[i] != neighbor[i]:
                modified_runways.append(i)
        
        if len(current_solution) != len(neighbor):
            neighbor_value = calculate_solution_value(
                neighbor, release_times, processing_times, waiting_times, penalties
            )
        else:
            value_change = calculate_incremental_change(
                current_solution, neighbor, modified_runways,
                release_times, processing_times, waiting_times, penalties
            )
            neighbor_value = current_value + value_change
        
        if neighbor_value < current_value:
            current_solution = neighbor
            current_value = neighbor_value
            k = 0
            no_improvement_count = 0
            
            if current_value < best_value:
                best_solution = copy.deepcopy(current_solution)
                best_value = current_value
        else:
            k += 1
            
            if k >= len(neighborhood_functions):
                k = 0
                no_improvement_count += 1
        
        iteration += 1
    
    elapsed_time = time.time() - start_time
    if elapsed_time < time_limit / 2 and iteration < max_iterations / 2:
        remaining_time = time_limit - elapsed_time
        remaining_iterations = max_iterations - iteration
        
        for _ in range(min(100, remaining_iterations)):
            if time.time() - start_time > time_limit:
                break
                
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
    
    stagnation_count = 0
    max_stagnation = 5 

    while iteration < max_iterations:
        if iteration % 50 == 0 and time.time() - start_time > time_limit:
             break

        improvement_found_in_cycle = False
        k = 0
        while k < len(neighborhood_functions):
            if time.time() - start_time > time_limit:
                break

            neighborhood_function = neighborhood_functions[k]
            
            improvement_found_in_neighborhood = False
            for _ in range(max_neighbors_per_iter):
                if time.time() - start_time > time_limit:
                    break

                result = neighborhood_function(
                    current_solution, release_times, processing_times, waiting_times, penalties
                )

                if result is None:
                    continue

                move_info, delta_penalty = result
                
                if delta_penalty < -1e-9:
                    move_type = move_info["type"]
                    if move_type == "swap":
                        r_idx = move_info["runway_idx"]
                        pos1 = move_info["pos1"]
                        pos2 = move_info["pos2"]
                        flight1 = current_solution[r_idx][pos1]
                        flight2 = current_solution[r_idx][pos2]
                        current_solution[r_idx][pos1] = flight2
                        current_solution[r_idx][pos2] = flight1
                    elif move_type == "move":
                        r_idx = move_info["runway_idx"]
                        from_pos = move_info["from_pos"]
                        to_pos = move_info["to_pos"]
                        flight = current_solution[r_idx].pop(from_pos)
                        current_solution[r_idx].insert(to_pos, flight)
                    elif move_type == "swap_between":
                        r1_idx = move_info["runway1_idx"]
                        r2_idx = move_info["runway2_idx"]
                        pos1 = move_info["pos1"]
                        pos2 = move_info["pos2"]
                        current_solution[r1_idx][pos1], current_solution[r2_idx][pos2] = \
                            current_solution[r2_idx][pos2], current_solution[r1_idx][pos1]
                    elif move_type == "reinsert":
                        s_idx = move_info["source_runway_idx"]
                        t_idx = move_info["target_runway_idx"]
                        s_pos = move_info["source_pos"]
                        t_pos = move_info["target_pos"]
                        flight_to_move = current_solution[s_idx].pop(s_pos)
                        current_solution[t_idx].insert(t_pos, flight_to_move)
                    elif move_type == "reinsert_delayed":
                        s_idx = move_info["source_runway_idx"]
                        t_idx = move_info["target_runway_idx"]
                        s_pos = move_info["source_pos"]
                        t_pos = move_info["target_pos"]
                        flight_to_move = current_solution[s_idx].pop(s_pos)
                        current_solution[t_idx].insert(t_pos, flight_to_move)
                    else:
                         pass

                    current_value += delta_penalty
                    improvement_found_in_neighborhood = True
                    improvement_found_in_cycle = True
                    stagnation_count = 0

                    if current_value < best_value:
                        best_solution = copy.deepcopy(current_solution)
                        best_value = current_value
                    
                    break 
            
            if improvement_found_in_neighborhood:
                k = 0 
            else:
                k += 1

        if not improvement_found_in_cycle:
             stagnation_count += 1
             if stagnation_count >= max_stagnation:
                  break

        iteration += 1

    return best_solution