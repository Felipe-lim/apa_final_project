import time
import glob
import os
import re
import copy
import csv
import io
import math

from algorithms.greedy import combined_heuristic, grasp_construction
from algorithms.vnd import vnd_algorithm, first_improvement_vnd
from algorithms.ils import iterated_local_search
from algorithms.neighborhood import (
    swap_flights, move_flight, swap_flights_between_runways, 
    reinsert_flight, reinsert_delayed_flight,
    shake_solution
)
from utils.file_handler import parse_input_file
from utils.metrics import calculate_solution_value, calculate_gap

EXPECTED_RESULTS = {
    "n3m10A": (7483, "opt"), "n3m10B": (1277, "opt"), "n3m10C": (2088, "opt"),
    "n3m10D": (322, "opt"), "n3m10E": (3343, "opt"), "n3m20A": (8280, "LB"),
    "n3m20B": (1820, "LB"), "n3m20C": (855, "LB"), "n3m20D": (4357, "opt"),
    "n3m20E": (3798, "opt"), "n3m40A": (112, "LB"), "n3m40B": (880, "LB"),
    "n3m40C": (1962, "LB"), "n3m40D": (263, "LB"), "n3m40E": (1192, "LB"),
}

def run_greedy(instance_data):
  rt, pt, wt, p, nr = instance_data
  start_time = time.time()
  schedule = combined_heuristic(rt, pt, wt, p, nr)
  penalty = calculate_solution_value(schedule, rt, pt, wt, p)
  exec_time = time.time() - start_time
  return penalty, exec_time, schedule

def run_vnd(instance_data, vnd_iter=1000, vnd_time=5.0):
  rt, pt, wt, p, nr = instance_data
  start_time = time.time()
  initial_schedule = combined_heuristic(rt, pt, wt, p, nr)
  neighborhoods = [
    swap_flights, move_flight, swap_flights_between_runways, 
    reinsert_flight, reinsert_delayed_flight
  ]
  final_schedule = first_improvement_vnd(
    initial_schedule, rt, pt, wt, p, neighborhoods,
    max_iterations=vnd_iter, 
    time_limit=vnd_time,
  )
  final_penalty = calculate_solution_value(final_schedule, rt, pt, wt, p)
  exec_time = time.time() - start_time
  return final_penalty, exec_time, final_schedule

def run_ils(instance_data, ils_iter=20, ils_time=30.0, perturb_intensity=10, 
            grasp_alpha=0.3, max_neighbors_ls=50, time_limit_ls=1.5):
  rt, pt, wt, p, nr = instance_data
  start_time = time.time()
  vnd_neighborhoods = [
    swap_flights, move_flight, swap_flights_between_runways,
    reinsert_flight, reinsert_delayed_flight
  ]
  constructive_func = lambda r_t, p_t, w_t, pen, n_r: grasp_construction(r_t, p_t, w_t, pen, n_r, alpha=grasp_alpha)
  max_iterations_ls = 1000

  best_schedule, final_penalty = iterated_local_search(
    rt, pt, wt, p, nr,
    constructive_heuristic=constructive_func,
    local_search=first_improvement_vnd,
    perturbation=shake_solution,
    neighborhoods_for_ls=vnd_neighborhoods,
    max_iterations_ils=ils_iter,
    max_time_ils=ils_time,
    perturbation_intensity=perturb_intensity,
    max_iterations_ls=max_iterations_ls,
    time_limit_ls=time_limit_ls
  )
  exec_time = time.time() - start_time
  return final_penalty, exec_time, best_schedule

def run_grasp(instance_data, grasp_iter=20, grasp_alpha=0.3, 
              vnd_iter=1000, vnd_time=5.0, max_neighbors_vnd=50):
  rt, pt, wt, p, nr = instance_data
  start_time = time.time()
  best_overall_penalty = float('inf')
  best_overall_schedule = None

  neighborhoods = [
    swap_flights, move_flight, swap_flights_between_runways,
    reinsert_flight, reinsert_delayed_flight
  ]

  for i in range(grasp_iter):
    if time.time() - start_time > PARAMS['GRASP'].get('grasp_time', 60.0):
      print(f"GRASP time limit reached at iteration {i}")
      break
      
    constructed_schedule = grasp_construction(rt, pt, wt, p, nr, alpha=grasp_alpha)
    
    improved_schedule = first_improvement_vnd(
      constructed_schedule, rt, pt, wt, p, neighborhoods,
      max_iterations=vnd_iter, 
      time_limit=vnd_time,
      max_neighbors_per_iter=max_neighbors_vnd
    )
    current_penalty = calculate_solution_value(improved_schedule, rt, pt, wt, p)

    if current_penalty < best_overall_penalty:
      best_overall_penalty = current_penalty
      best_overall_schedule = copy.deepcopy(improved_schedule)

  exec_time = time.time() - start_time
  if best_overall_penalty == float('inf'):
    best_overall_penalty = None 

  return best_overall_penalty, exec_time, best_overall_schedule

if __name__ == "__main__":
  test_dir = "tests"
  instance_pattern = os.path.join(test_dir, "n*.txt")
  instance_files = sorted(glob.glob(instance_pattern))

  PARAMS = {
    'Greedy': {'iter': 1},
    'VND': {'iter': 1, 'vnd_iter': 1000, 'vnd_time': 2, 'max_neighbors_vnd': 25}, 
    'ILS': {'iter': 25,          
            'ils_time': 3.5,
            'perturb': 20,
            'alpha': 0.5, 
            'max_neighbors_ls': 35,
            'time_limit_ls': 2.5},
    'GRASP': {'iter': 50,       
              'grasp_alpha': 0.5,
              'vnd_iter': 1000,
              'vnd_time': 2.0,
              'max_neighbors_vnd': 30,
              'grasp_time': 2.5}
  }
  ALGORITHMS_TO_RUN = list(PARAMS.keys())

  results_list = []

  if not instance_files:
    print(f"No instance files found in {test_dir} matching pattern {instance_pattern}")
  else:
    print(f"Found {len(instance_files)} instances in {test_dir}.")
    print("Running evaluations (using First Improvement VND for ILS/GRASP/VND)...")
    print(f"Algorithms: {', '.join(ALGORITHMS_TO_RUN)}")
    print("Parameters:", PARAMS)
    print("=" * 60)

    for instance_path in instance_files:
      instance_name = os.path.splitext(os.path.basename(instance_path))[0]
      
      if instance_name.startswith("n5m"):
        continue

      print(f"--- Instance: {instance_name} ---")
      instance_data = None
      try:
        with open(instance_path, 'r') as f:
          num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(f)
        instance_data = (release_times, processing_times, waiting_times, penalties, num_runways)
      except Exception as e:
        print(f"  ERROR parsing: {e}")
        print("=" * 60)
        continue
      ref_value, ref_type = EXPECTED_RESULTS.get(instance_name, (None, None))
      print(f"  Reference: {ref_value} ({ref_type if ref_type else 'N/A'})")
      instance_run_results = {}
      run_error = False
      for algo_name in ALGORITHMS_TO_RUN:
        try:
          penalty, ex_time, schedule = float('nan'), float('nan'), None
          if algo_name == 'Greedy':
            penalty, ex_time, schedule = run_greedy(instance_data)
          elif algo_name == 'VND':
            penalty, ex_time, schedule = run_vnd(instance_data, 
                                     vnd_iter=PARAMS['VND']['vnd_iter'], 
                                     vnd_time=PARAMS['VND']['vnd_time'])
          elif algo_name == 'ILS':
            penalty, ex_time, schedule = run_ils(instance_data, 
                                     ils_iter=PARAMS['ILS']['iter'], 
                                     ils_time=PARAMS['ILS']['ils_time'],
                                     perturb_intensity=PARAMS['ILS']['perturb'], 
                                     grasp_alpha=PARAMS['ILS']['alpha'],
                                     time_limit_ls=PARAMS['ILS']['time_limit_ls'],
                                     )
          elif algo_name == 'GRASP':
            penalty, ex_time, schedule = run_grasp(instance_data, 
                                      grasp_iter=PARAMS['GRASP']['iter'], 
                                      grasp_alpha=PARAMS['GRASP']['grasp_alpha'],
                                      vnd_iter=PARAMS['GRASP']['vnd_iter'], 
                                      vnd_time=PARAMS['GRASP']['vnd_time'],
                                      max_neighbors_vnd=PARAMS['GRASP']['max_neighbors_vnd'])
          
          gap = float('nan')
          if penalty is not None and not math.isnan(penalty) and ref_value is not None:
            gap = calculate_gap(penalty, ref_value)
          
          instance_run_results[algo_name] = {
            'Penalty': penalty, 
            'Time': ex_time, 
            'GAP': gap,
            'Schedule': schedule
          }
          
          results_list.append({
            'Instance': instance_name, 'Algorithm': algo_name, 'Penalty': penalty,
            'Time': ex_time, 'Ref_Value': ref_value, 'Ref_Type': ref_type, 'GAP': gap
          })

        except Exception as e:
          print(f"  ERROR running {algo_name}: {e}")
          import traceback
          traceback.print_exc()
          instance_run_results[algo_name] = {
            'Penalty': 'ERROR', 
            'Time': 'ERROR', 
            'GAP': 'ERROR',
            'Schedule': None
          }
          run_error = True
          results_list.append({
            'Instance': instance_name, 'Algorithm': algo_name, 'Penalty': None,
            'Time': None, 'Ref_Value': ref_value, 'Ref_Type': ref_type, 'GAP': None
          })
      
      print("-" * 50)
      print(f"  {'Algorithm':<10} {'Penalty':>12} {'Time (s)':>10} {'GAP (%)':>10}")
      print("-" * 50)
      for algo_name in ALGORITHMS_TO_RUN:
        res = instance_run_results.get(algo_name, {})
        penalty_val = res.get('Penalty', 'N/A')
        time_val = res.get('Time', 'N/A')
        gap_val = res.get('GAP')

        pen_str = f"{penalty_val:>12.2f}" if isinstance(penalty_val, (int, float)) and not math.isnan(penalty_val) else f"{str(penalty_val):>12}"
        time_str = f"{time_val:>10.4f}" if isinstance(time_val, (int, float)) and not math.isnan(time_val) else f"{str(time_val):>10}"
        
        gap_str = "N/A"
        if isinstance(gap_val, (int, float)) and not math.isnan(gap_val):
          if gap_val == float('inf'):
            gap_str = "inf"
          else:
            gap_str = f"{gap_val:.2f}"
        
        print(f"  {algo_name:<10} {pen_str} {time_str} {gap_str:>10}")
      print("-" * 50)
      
      print("  Final Schedules:")
      for algo_name in ALGORITHMS_TO_RUN:
        res = instance_run_results.get(algo_name, {})
        schedule = res.get('Schedule')
        print(f"    {algo_name}:")
        if schedule is None:
          print("      Not available (Error during run or not generated)")
        elif not schedule:
          print("      [] (Empty schedule)")
        else:
          for r_idx, runway in enumerate(schedule):
            runway_str = str(runway)
            if len(runway_str) > 80:
              runway_str = runway_str[:77] + "..."
            print(f"      Runway {r_idx}: {runway_str}")
      print("-" * 50)

    print("-" * 60)
    print("Evaluation complete.")
    print("-" * 60)
    
    if results_list:
      print("\nFinal Results (CSV Format - Excluding LB=0 Instances):")
      output = io.StringIO()
      fieldnames = ['Instance', 'Algorithm', 'Penalty', 'Time', 'Ref_Value', 'Ref_Type', 'GAP']
      writer = csv.DictWriter(output, fieldnames=fieldnames)
      writer.writeheader()
      valid_results = [row for row in results_list if 
                       (row['Penalty'] is not None or row['Time'] is not None) and 
                       (row['Ref_Value'] is None or row['Ref_Value'] != 0)] 
      writer.writerows(valid_results)
      print(output.getvalue())
    else:
      print("No results generated.")

  print("Script finished.")