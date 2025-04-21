import time
import copy
from typing import List, Callable, Tuple

from algorithms.vnd import first_improvement_vnd
from algorithms.neighborhood import shake_solution
from utils.metrics import calculate_solution_value

def iterated_local_search(
    release_times: List[int],
    processing_times: List[int],
    waiting_times: List[List[int]],
    penalties: List[int],
    num_runways: int,
    constructive_heuristic: Callable,
    local_search: Callable,
    perturbation: Callable,
    neighborhoods_for_ls: List[Callable],
    max_iterations_ils: int = 100,
    max_time_ils: float = 10.0,
    perturbation_intensity: int = 5,
    max_iterations_ls: int = 500,
    time_limit_ls: float = 1.0
) -> Tuple[List[List[int]], float]:
    start_time_total = time.time()

    current_schedule = constructive_heuristic(
        release_times, processing_times, waiting_times, penalties, num_runways
    )

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

        perturbed_schedule = perturbation(
            best_schedule,
            release_times, processing_times, waiting_times, penalties,
            intensity=perturbation_intensity
        )

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

        if new_penalty < best_penalty:
            best_schedule = copy.deepcopy(new_schedule)
            best_penalty = new_penalty

        if (time.time() - start_time_total) >= max_time_ils:
            break

    return best_schedule, best_penalty