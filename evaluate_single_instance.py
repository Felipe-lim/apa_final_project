import time

from algorithms.greedy import combined_heuristic
from algorithms.vnd import vnd_algorithm
from algorithms.neighborhood import swap_flights, move_flight, swap_flights_between_runways
from utils.file_handler import parse_input_file
from utils.metrics import calculate_solution_value

def evaluate_instance(instance_path):
  """Loads an instance, solves it using greedy + VND, and returns the penalty."""
  # Parse instance using parse_input_file
  try:
    with open(instance_path, 'r') as f:
      num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(f)
  except FileNotFoundError:
    print(f"Error: Instance file not found at {instance_path}")
    return None
  except ValueError as e:
    print(f"Error parsing instance file {instance_path}: {e}")
    return None

  print(f"Evaluating instance: {instance_path}")
  print(f"Number of flights: {num_flights}")
  print(f"Number of runways: {num_runways}")
  
  # Run Greedy (using combined heuristic)
  print(f"\nRunning Combined Heuristic...")
  start_time_greedy = time.time()
  initial_schedule = combined_heuristic(
    release_times, processing_times, waiting_times, penalties, num_runways
  )
  # Calculate initial penalty using calculate_solution_value
  initial_penalty = calculate_solution_value(
    initial_schedule, release_times, processing_times, waiting_times, penalties
  )
  end_time_greedy = time.time()
  
  print(f"Initial Greedy Solution (Combined Heuristic):")
  print(f"  Penalty: {initial_penalty:.2f}")
  print(f"  Time: {end_time_greedy - start_time_greedy:.4f} seconds")
  
  # Define the list of neighborhood functions for VND
  neighborhoods = [swap_flights, move_flight, swap_flights_between_runways]

  # Run VND
  print(f"\nRunning VND...")
  start_time_vnd = time.time()
  final_schedule = vnd_algorithm(
    initial_schedule, 
    release_times, 
    processing_times, 
    waiting_times, 
    penalties, 
    neighborhoods,
    # Optional: Add parameters for max_iterations and time_limit
    # max_iterations=1000, 
    # time_limit=10.0 
  )
  end_time_vnd = time.time()

  # Calculate final penalty using calculate_solution_value
  final_penalty = calculate_solution_value(
    final_schedule, release_times, processing_times, waiting_times, penalties
  )
  
  print(f"\nFinal Solution (after VND):")
  print(f"  Penalty: {final_penalty:.2f}")
  print(f"  Time: {end_time_vnd - start_time_vnd:.4f} seconds")
  
  total_time = (end_time_greedy - start_time_greedy) + (end_time_vnd - start_time_vnd)
  print(f"\nTotal time: {total_time:.4f} seconds")

  # Optional: Print the final schedule
  # print("\nFinal Schedule:")
  # for r_idx, runway_schedule in enumerate(final_schedule):
  #   print(f"  Runway {r_idx + 1}: {runway_schedule}")
    
  return final_penalty

if __name__ == "__main__":
  instance_file = "example_instance.txt" 
  final_penalty_value = evaluate_instance(instance_file)
  if final_penalty_value is not None:
    print(f"\nObjective Function Value (Total Penalty) for {instance_file}: {final_penalty_value:.2f}") 