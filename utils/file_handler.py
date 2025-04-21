import io
from typing import List, Tuple

def parse_input_file(file) -> Tuple[int, int, List[int], List[int], List[int], List[List[int]]]:
    if hasattr(file, 'seek'):
        file.seek(0)
    
    lines = file.readlines()
    
    num_flights = int(lines[0].strip().split()[0])
    num_runways = int(lines[1].strip().split()[0])
    
    release_times = list(map(int, lines[3].strip().split()))
    processing_times = list(map(int, lines[4].strip().split()))
    penalties = list(map(int, lines[5].strip().split()))
    
    waiting_times = []
    for i in range(7, 7 + num_flights):
        if i < len(lines):
            row = list(map(int, lines[i].strip().split()))
            waiting_times.append(row)
    
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
