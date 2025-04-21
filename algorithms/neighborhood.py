import random
import copy
from typing import List, Tuple, Set, Optional, Dict, Any

def calculate_runway_penalty(runway: List[int], 
                             release_times: List[int], 
                             processing_times: List[int], 
                             waiting_times: List[List[int]], 
                             penalties: List[int]) -> float:
  runway_penalty = 0
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
    runway_penalty += delay * penalties[flight]
    current_time = start_time + processing_times[flight]
    prev_flight = flight
  return runway_penalty

def swap_flights(solution: List[List[int]], 
                 release_times: List[int], 
                 processing_times: List[int], 
                 waiting_times: List[List[int]], 
                 penalties: List[int]) -> Optional[Tuple[Dict[str, Any], float]]:
  eligible_runways = [i for i, r in enumerate(solution) if len(r) >= 2]
  if not eligible_runways:
    return None

  runway_idx = random.choice(eligible_runways)
  runway = solution[runway_idx]
  
  if len(runway) < 2:
      return None 

  swap_pos = random.randint(0, len(runway) - 2)

  original_penalty = calculate_runway_penalty(
      runway, release_times, processing_times, waiting_times, penalties
  )

  swapped_runway = runway[:]
  swapped_runway[swap_pos], swapped_runway[swap_pos + 1] = swapped_runway[swap_pos + 1], swapped_runway[swap_pos]

  new_penalty = calculate_runway_penalty(
      swapped_runway, release_times, processing_times, waiting_times, penalties
  )
  
  delta_penalty = new_penalty - original_penalty

  move_info = {
      "type": "swap", 
      "runway_idx": runway_idx, 
      "pos1": swap_pos, 
      "pos2": swap_pos + 1
  }
  return move_info, delta_penalty

def move_flight(solution: List[List[int]], 
               release_times: List[int], 
               processing_times: List[int], 
               waiting_times: List[List[int]], 
               penalties: List[int]) -> Optional[Tuple[Dict[str, Any], float]]:
  eligible_runways = [i for i, r in enumerate(solution) if len(r) >= 1]
  if not eligible_runways:
    return None

  runway_idx = random.choice(eligible_runways)
  runway = solution[runway_idx]

  if not runway:
       return None

  from_pos = random.randint(0, len(runway) - 1)
  flight_to_move = runway[from_pos]

  to_pos = random.randint(0, len(runway))

  original_penalty = calculate_runway_penalty(
      runway, release_times, processing_times, waiting_times, penalties
  )

  modified_runway = runway[:]
  flight = modified_runway.pop(from_pos)
  actual_insert_pos = to_pos
  if from_pos < to_pos:
      actual_insert_pos -= 1 
  modified_runway.insert(actual_insert_pos, flight)

  new_penalty = calculate_runway_penalty(
      modified_runway, release_times, processing_times, waiting_times, penalties
  )

  delta_penalty = new_penalty - original_penalty

  move_info = {
      "type": "move",
      "runway_idx": runway_idx,
      "from_pos": from_pos,
      "to_pos": actual_insert_pos
  }
  return move_info, delta_penalty

def swap_flights_between_runways(solution: List[List[int]], 
                                release_times: List[int], 
                                processing_times: List[int], 
                                waiting_times: List[List[int]], 
                                penalties: List[int]) -> Optional[Tuple[Dict[str, Any], float]]:
  non_empty_runways = [i for i, r in enumerate(solution) if len(r) >= 1]
  if len(non_empty_runways) < 2:
    return None

  runway1_idx, runway2_idx = random.sample(non_empty_runways, 2)
  runway1 = solution[runway1_idx]
  runway2 = solution[runway2_idx]

  flight1_pos = random.randint(0, len(runway1) - 1)
  flight2_pos = random.randint(0, len(runway2) - 1)

  original_penalty_r1 = calculate_runway_penalty(
      runway1, release_times, processing_times, waiting_times, penalties
  )
  original_penalty_r2 = calculate_runway_penalty(
      runway2, release_times, processing_times, waiting_times, penalties
  )

  temp_runway1 = runway1[:]
  temp_runway2 = runway2[:]
  temp_runway1[flight1_pos], temp_runway2[flight2_pos] = temp_runway2[flight2_pos], temp_runway1[flight1_pos]

  new_penalty_r1 = calculate_runway_penalty(
      temp_runway1, release_times, processing_times, waiting_times, penalties
  )
  new_penalty_r2 = calculate_runway_penalty(
      temp_runway2, release_times, processing_times, waiting_times, penalties
  )

  delta_penalty = (new_penalty_r1 + new_penalty_r2) - (original_penalty_r1 + original_penalty_r2)

  move_info = {
      "type": "swap_between",
      "runway1_idx": runway1_idx,
      "runway2_idx": runway2_idx,
      "pos1": flight1_pos,
      "pos2": flight2_pos
  }
  return move_info, delta_penalty

def reinsert_flight(solution: List[List[int]], 
                   release_times: List[int], 
                   processing_times: List[int], 
                   waiting_times: List[List[int]], 
                   penalties: List[int]) -> Optional[Tuple[Dict[str, Any], float]]:
  non_empty_runways = [i for i, r in enumerate(solution) if len(r) >= 1]
  if not non_empty_runways:
    return None
  
  source_runway_idx = random.choice(non_empty_runways)
  source_runway = solution[source_runway_idx]
  flight_pos = random.randint(0, len(source_runway) - 1)
  flight_to_move = source_runway[flight_pos]

  target_runway_idx = random.randint(0, len(solution) - 1)
  target_runway = solution[target_runway_idx]

  insert_pos = random.randint(0, len(target_runway))

  original_penalty_source = calculate_runway_penalty(
      source_runway, release_times, processing_times, waiting_times, penalties
  )
  original_penalty_target = 0
  if source_runway_idx != target_runway_idx:
      original_penalty_target = calculate_runway_penalty(
          target_runway, release_times, processing_times, waiting_times, penalties
      )

  temp_source_runway = source_runway[:]
  temp_source_runway.pop(flight_pos)
  
  temp_target_runway = target_runway[:]
  if source_runway_idx == target_runway_idx:
      temp_target_runway = temp_source_runway
      actual_insert_pos = insert_pos
      if flight_pos < insert_pos:
           actual_insert_pos -= 1
      temp_target_runway.insert(actual_insert_pos, flight_to_move)
      final_insert_pos = actual_insert_pos
  else:
      temp_target_runway.insert(insert_pos, flight_to_move)
      final_insert_pos = insert_pos

  new_penalty_source = calculate_runway_penalty(
      temp_source_runway, release_times, processing_times, waiting_times, penalties
  )
  new_penalty_target = 0
  if source_runway_idx != target_runway_idx:
       new_penalty_target = calculate_runway_penalty(
           temp_target_runway, release_times, processing_times, waiting_times, penalties
       )
  else:
       new_penalty_target = new_penalty_source

  delta_penalty = (new_penalty_source + new_penalty_target) - (original_penalty_source + original_penalty_target)
  if source_runway_idx == target_runway_idx:
       delta_penalty = new_penalty_target - original_penalty_source

  move_info = {
      "type": "reinsert",
      "source_runway_idx": source_runway_idx,
      "target_runway_idx": target_runway_idx,
      "source_pos": flight_pos,
      "target_pos": final_insert_pos
  }
  return move_info, delta_penalty

def reinsert_delayed_flight(solution: List[List[int]], 
                           release_times: List[int], 
                           processing_times: List[int], 
                           waiting_times: List[List[int]], 
                           penalties: List[int]) -> Optional[Tuple[Dict[str, Any], float]]:
    if not solution or all(len(runway) == 0 for runway in solution):
        return None
    
    flight_delays = {}
    flight_penalties = {}
    flight_runways = {}
    flight_positions = {}
    worst_flight = -1
    max_penalty = -1
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        for pos, flight in enumerate(runway):
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            delay = max(0, start_time - release_times[flight])
            penalty = delay * penalties[flight]
            flight_delays[flight] = delay
            flight_penalties[flight] = penalty
            flight_runways[flight] = runway_idx
            flight_positions[flight] = pos
            if penalty > max_penalty:
                 max_penalty = penalty
                 worst_flight = flight
            current_time = start_time + processing_times[flight]
            prev_flight = flight
    
    if worst_flight == -1 or max_penalty <= 0:
        return None 

    source_runway_idx = flight_runways[worst_flight]
    source_pos = flight_positions[worst_flight]
    flight_to_move = solution[source_runway_idx][source_pos]

    original_penalty_source = calculate_runway_penalty(
        solution[source_runway_idx], release_times, processing_times, waiting_times, penalties
    )

    best_move_delta = float('inf')
    best_target_runway_idx = -1
    best_target_pos = -1

    for target_runway_idx in range(len(solution)):
        target_runway_original = solution[target_runway_idx]
        original_penalty_target = 0
        if source_runway_idx != target_runway_idx:
            original_penalty_target = calculate_runway_penalty(
                target_runway_original, release_times, processing_times, waiting_times, penalties
            )
        
        for insert_pos in range(len(target_runway_original) + 1):
            if source_runway_idx == target_runway_idx and source_pos == insert_pos:
                continue

            temp_source_runway = solution[source_runway_idx][:]
            temp_source_runway.pop(source_pos)

            temp_target_runway = target_runway_original[:]
            actual_insert_pos = insert_pos
            if source_runway_idx == target_runway_idx:
                 temp_target_runway = temp_source_runway
                 if source_pos < insert_pos:
                      actual_insert_pos -= 1
                 temp_target_runway.insert(actual_insert_pos, flight_to_move)
            else:
                 temp_target_runway.insert(insert_pos, flight_to_move)

            new_penalty_source = calculate_runway_penalty(
                temp_source_runway, release_times, processing_times, waiting_times, penalties
            )
            new_penalty_target = calculate_runway_penalty(
                temp_target_runway, release_times, processing_times, waiting_times, penalties
            )
            
            current_delta = 0
            if source_runway_idx == target_runway_idx:
                 current_delta = new_penalty_target - original_penalty_source
            else:
                 current_delta = (new_penalty_source + new_penalty_target) - \
                                 (original_penalty_source + original_penalty_target)
            
            if current_delta < best_move_delta:
                best_move_delta = current_delta
                best_target_runway_idx = target_runway_idx
                best_target_pos = actual_insert_pos
    
    if best_target_runway_idx == -1 or best_move_delta >= 0:
        return None

    move_info = {
        "type": "reinsert_delayed",
        "source_runway_idx": source_runway_idx,
        "target_runway_idx": best_target_runway_idx,
        "source_pos": source_pos,
        "target_pos": best_target_pos
    }
    return move_info, best_move_delta

def shake_solution(solution: List[List[int]], 
                  release_times: List[int], 
                  processing_times: List[int], 
                  waiting_times: List[List[int]], 
                  penalties: List[int],
                  intensity: int = 3) -> List[List[int]]:
    new_solution = copy.deepcopy(solution)
    
    neighborhoods = [
        swap_flights_between_runways,
        reinsert_flight,
        reinsert_delayed_flight
    ]
    
    for _ in range(intensity):
        if not neighborhoods:
             break
        neighborhood_function = random.choice(neighborhoods)
        
        result = neighborhood_function(
            new_solution, release_times, processing_times, waiting_times, penalties
        )
        if isinstance(result, list):
             new_solution = result

    return new_solution

def optimize_runway_balance(solution: List[List[int]], 
                           release_times: List[int], 
                           processing_times: List[int], 
                           waiting_times: List[List[int]], 
                           penalties: List[int]) -> List[List[int]]:
    new_solution = copy.deepcopy(solution)
    
    total_flights = sum(len(runway) for runway in new_solution)
    avg_flights = total_flights / len(new_solution)
    
    runway_loads = [(i, len(runway)) for i, runway in enumerate(new_solution)]
    runway_loads.sort(key=lambda x: x[1], reverse=True)
    
    if len(runway_loads) <= 1 or runway_loads[0][1] - runway_loads[-1][1] <= 1:
        return new_solution
    
    overloaded_idx = runway_loads[0][0]
    underloaded_idx = runway_loads[-1][0]
    
    overloaded_runway = new_solution[overloaded_idx]
    underloaded_runway = new_solution[underloaded_idx]
    
    best_penalty_increase = float('inf')
    best_flight_pos = -1
    best_insert_pos = -1
    
    current_penalty = 0
    
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
    
    for flight_pos, flight in enumerate(overloaded_runway):
        for insert_pos in range(len(underloaded_runway) + 1):
            test_overloaded = overloaded_runway.copy()
            test_underloaded = underloaded_runway.copy()
            
            flight_obj = test_overloaded.pop(flight_pos)
            test_underloaded.insert(insert_pos, flight_obj)
            
            new_penalty = 0
            
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
            
            penalty_increase = new_penalty - current_penalty
            
            if penalty_increase < best_penalty_increase:
                best_penalty_increase = penalty_increase
                best_flight_pos = flight_pos
                best_insert_pos = insert_pos
    
    if best_flight_pos != -1:
        flight_obj = new_solution[overloaded_idx].pop(best_flight_pos)
        new_solution[underloaded_idx].insert(best_insert_pos, flight_obj)
    
    return new_solution