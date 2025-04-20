import time
import copy
import random
from typing import List, Callable, Tuple

from algorithms.vnd import vnd_algorithm # Or any other local search
from algorithms.neighborhood import shake_solution # Or any other perturbation
from utils.metrics import calculate_solution_value

def iterated_local_search(
    # Instance data
    release_times: List[int],
    processing_times: List[int],
    waiting_times: List[List[int]],
    penalties: List[int],
    num_runways: int,
    # Algorithm components
    constructive_heuristic: Callable,
    local_search: Callable,
    perturbation: Callable,
    neighborhoods_for_ls: List[Callable],
    # Control parameters
    max_iterations_ils: int = 100, # Iterations of ILS (Perturb + Local Search)
    max_time_ils: float = 10.0,  # Max total time in seconds
    perturbation_intensity: int = 5, # Intensity for shake_solution
    # Optional: Parameters for the local search within ILS
    max_iterations_ls: int = 500, # Iterations for inner VND
    time_limit_ls: float = 1.0    # Time limit for inner VND
) -> Tuple[List[List[int]], float]:
    """
    Performs Iterated Local Search (ILS).

    Args:
        release_times, processing_times, waiting_times, penalties, num_runways: Instance data.
        constructive_heuristic: Function to generate initial solution.
        local_search: Function for the local search phase (e.g., vnd_algorithm).
        perturbation: Function to shake the solution (e.g., shake_solution).
        neighborhoods_for_ls: List of neighborhoods for the local search function.
        max_iterations_ils: Max iterations for the main ILS loop.
        max_time_ils: Overall time limit for ILS.
        perturbation_intensity: Strength of the perturbation.
        max_iterations_ls: Max iterations for the inner local search.
        time_limit_ls: Time limit for the inner local search.

    Returns:
        Tuple containing the best solution found (List[List[int]]) and its penalty (float).
    """
    start_time_total = time.time()

    # 1. Initial Solution
    current_schedule = constructive_heuristic(
        release_times, processing_times, waiting_times, penalties, num_runways
    )

    # 2. Initial Local Search
    best_schedule = local_search(
        current_schedule,
        release_times, processing_times, waiting_times, penalties,
        neighborhoods_for_ls,
        max_iterations=max_iterations_ls,
        time_limit=time_limit_ls
    )
    best_penalty = calculate_solution_value(
        best_schedule, release_times, processing_times, waiting_times, penalties
    )

    iteration = 0
    while iteration < max_iterations_ils and (time.time() - start_time_total) < max_time_ils:
        iteration += 1

        # 3. Perturbation
        perturbed_schedule = perturbation(
            best_schedule, # Perturb the current best solution
            release_times, processing_times, waiting_times, penalties,
            intensity=perturbation_intensity
        )

        # 4. Local Search on Perturbed Solution
        new_schedule = local_search(
            perturbed_schedule,
            release_times, processing_times, waiting_times, penalties,
            neighborhoods_for_ls,
            max_iterations=max_iterations_ls,
            time_limit=time_limit_ls
        )
        new_penalty = calculate_solution_value(
            new_schedule, release_times, processing_times, waiting_times, penalties
        )

        # 5. Acceptance Criterion (Best Improvement)
        if new_penalty < best_penalty:
            best_schedule = copy.deepcopy(new_schedule)
            best_penalty = new_penalty
            # print(f"ILS Iter {iteration}: New best found: {best_penalty:.2f}") # Optional debug print
        
        # Optional: Check time limit again inside loop for finer control
        if (time.time() - start_time_total) >= max_time_ils:
            break

    return best_schedule, best_penalty 