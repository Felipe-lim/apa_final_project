import io
from typing import List, Tuple

def parse_input_file(file) -> Tuple[int, int, List[int], List[int], List[int], List[List[int]]]:
    """
    Parse input file with problem instance data.
    
    Args:
        file: File object containing the problem instance
        
    Returns:
        Tuple containing:
        - number of flights
        - number of runways
        - release times
        - processing times
        - penalties
        - waiting times matrix
    """
    # Reset file pointer to the beginning
    if hasattr(file, 'seek'):
        file.seek(0)
    
    lines = file.readlines()
    
    # Extract number of flights and runways
    num_flights = int(lines[0].strip().split()[0])
    num_runways = int(lines[1].strip().split()[0])
    
    # Extract release times, processing times, and penalties
    release_times = list(map(int, lines[3].strip().split()))
    processing_times = list(map(int, lines[4].strip().split()))
    penalties = list(map(int, lines[5].strip().split()))
    
    # Extract waiting times matrix
    waiting_times = []
    for i in range(7, 7 + num_flights):
        if i < len(lines):
            row = list(map(int, lines[i].strip().split()))
            waiting_times.append(row)
    
    # Validate data
    if len(release_times) != num_flights:
        raise ValueError(f"Expected {num_flights} release times, got {len(release_times)}")
    if len(processing_times) != num_flights:
        raise ValueError(f"Expected {num_flights} processing times, got {len(processing_times)}")
    if len(penalties) != num_flights:
        raise ValueError(f"Expected {num_flights} penalties, got {len(penalties)}")
    if len(waiting_times) != num_flights:
        raise ValueError(f"Expected {num_flights} rows in waiting times matrix, got {len(waiting_times)}")
    for i, row in enumerate(waiting_times):
        if len(row) != num_flights:
            raise ValueError(f"Expected {num_flights} columns in row {i} of waiting times matrix, got {len(row)}")
    
    return num_flights, num_runways, release_times, processing_times, penalties, waiting_times

def write_solution_file(file, solution: List[List[int]], solution_value: float) -> None:
    """
    Write solution to file in the required format.
    
    Args:
        file: File object to write to
        solution: Assignment of flights to runways
        solution_value: Value of the solution (total penalty)
    """
    # Write solution value
    file.write(f"{solution_value}\n")
    
    # Write runway assignments (convert to 1-based indices)
    for runway_idx, runway in enumerate(solution):
        # Convert to 1-based indices
        runway_flights = [str(flight + 1) for flight in runway]
        file.write(" ".join(runway_flights) + "\n")

def parse_optimal_solution_file(file) -> Tuple[float, List[List[int]]]:
    """
    Parse a file containing an optimal solution.
    
    Args:
        file: File object containing the optimal solution
        
    Returns:
        Tuple containing:
        - solution value
        - assignment of flights to runways
    """
    lines = file.readlines()
    
    # Extract solution value
    solution_value = float(lines[0].strip())
    
    # Extract runway assignments
    solution = []
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line:
            # Convert from 1-based to 0-based indices
            runway = [int(x) - 1 for x in line.split()]
            solution.append(runway)
    
    return solution_value, solution
