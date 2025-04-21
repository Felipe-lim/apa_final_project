from typing import List, Callable, Tuple
import copy
import time
from utils.metrics import calculate_solution_value

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