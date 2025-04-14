import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import altair as alt
import pandas as pd

def plot_runway_schedule(solution: List[List[int]], 
                         release_times: List[int], 
                         processing_times: List[int], 
                         penalties: List[int],
                         num_runways: int,
                         waiting_times: List[List[int]] = None) -> plt.Figure:
    """
    Plot a visualization of flight scheduling on runways.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        penalties: Penalties per unit time for each flight
        num_runways: Number of runways
        
    Returns:
        Matplotlib figure object
    """
    # Calculate start and end times for each flight
    flight_schedule = []
    total_penalty = 0
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                # Add waiting time from previous flight to current flight
                if waiting_times is not None:
                    min_start_time = current_time + waiting_times[prev_flight][flight]
                    start_time = max(min_start_time, release_times[flight])
                else:
                    start_time = max(current_time, release_times[flight])
            
            # Calculate end time
            end_time = start_time + processing_times[flight]
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            total_penalty += flight_penalty
            
            # Store schedule information
            flight_schedule.append({
                'runway': runway_idx + 1,
                'flight': flight + 1,  # Convert to 1-based indexing for display
                'release_time': release_times[flight],
                'start_time': start_time,
                'end_time': end_time,
                'processing_time': processing_times[flight],
                'delay': delay,
                'penalty': flight_penalty
            })
            
            # Update current time for next flight
            current_time = end_time
            prev_flight = flight
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define a color map for flights
    colors = plt.cm.tab10.colors
    
    # Plot each flight as a rectangle
    for flight_info in flight_schedule:
        runway = flight_info['runway']
        flight = flight_info['flight']
        start = flight_info['start_time']
        duration = flight_info['processing_time']
        
        # Plot the flight block
        rect = plt.Rectangle(
            (start, runway - 0.4),
            duration,
            0.8,
            facecolor=colors[flight % len(colors)],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add flight number label
        plt.text(
            start + duration / 2,
            runway,
            f"F{flight}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            fontweight='bold'
        )
        
        # Add release time marker
        release_time = flight_info['release_time']
        plt.axvline(x=release_time, color='gray', linestyle='--', alpha=0.3)
        
        # If delayed, add a marker
        if flight_info['delay'] > 0:
            plt.plot(
                [release_time, start],
                [runway, runway],
                'r-',
                linewidth=2,
                alpha=0.7
            )
            plt.text(
                (release_time + start) / 2,
                runway + 0.3,
                f"Delay: {flight_info['delay']}",
                color='red',
                fontsize=8,
                horizontalalignment='center'
            )
    
    # Set axis limits and labels
    ax.set_ylim(0.5, num_runways + 0.5)
    ax.set_xlim(0, max([info['end_time'] for info in flight_schedule]) * 1.1)
    ax.set_yticks(list(range(1, num_runways + 1)))
    ax.set_yticklabels([f"Runway {i}" for i in range(1, num_runways + 1)])
    ax.set_xlabel('Time')
    ax.set_title(f'Flight Schedule (Total Penalty: {total_penalty})')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_interactive_schedule(solution: List[List[int]], 
                                 release_times: List[int], 
                                 processing_times: List[int], 
                                 penalties: List[int],
                                 waiting_times: List[List[int]]) -> alt.Chart:
    """
    Generate an interactive Altair visualization of the flight schedule.
    
    Args:
        solution: Assignment of flights to runways
        release_times: Release times for all flights
        processing_times: Processing times for all flights
        penalties: Penalties per unit time for each flight
        waiting_times: Matrix of required waiting times between consecutive flights
        
    Returns:
        Altair chart object
    """
    # Calculate start and end times for each flight
    flight_data = []
    
    for runway_idx, runway in enumerate(solution):
        current_time = 0
        prev_flight = None
        
        for flight in runway:
            # Calculate start time
            if prev_flight is None:
                start_time = max(current_time, release_times[flight])
            else:
                min_start_time = current_time + waiting_times[prev_flight][flight]
                start_time = max(min_start_time, release_times[flight])
            
            # Calculate end time
            end_time = start_time + processing_times[flight]
            
            # Calculate delay and penalty
            delay = max(0, start_time - release_times[flight])
            flight_penalty = delay * penalties[flight]
            
            # Store schedule information
            flight_data.append({
                'Runway': f"Runway {runway_idx + 1}",
                'Flight': f"Flight {flight + 1}",
                'Start': start_time,
                'End': end_time,
                'Duration': processing_times[flight],
                'Release Time': release_times[flight],
                'Delay': delay,
                'Penalty': flight_penalty
            })
            
            # Update current time for next flight
            current_time = end_time
            prev_flight = flight
    
    # Convert to DataFrame
    df = pd.DataFrame(flight_data)
    
    # Create the Altair chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Start:Q', title='Time'),
        x2=alt.X2('End:Q'),
        y=alt.Y('Runway:N', title=''),
        color=alt.Color('Flight:N', legend=alt.Legend(title='Flights')),
        tooltip=['Flight', 'Start', 'End', 'Duration', 'Release Time', 'Delay', 'Penalty']
    ).properties(
        width=800,
        height=400,
        title='Flight Schedule'
    ).interactive()
    
    # Add release time markers
    release_markers = alt.Chart(df).mark_rule(color='gray', strokeDash=[3, 3]).encode(
        x='Release Time:Q',
        y=alt.Y('Runway:N'),
        tooltip=['Flight', 'Release Time']
    )
    
    return chart + release_markers

def plot_comparative_results(constructive_results: List[float], 
                             vnd_results: List[float],
                             optimal_value: float = 0) -> plt.Figure:
    """
    Plot comparative results between constructive algorithm and VND.
    
    Args:
        constructive_results: List of solution values from constructive algorithm
        vnd_results: List of solution values from VND
        optimal_value: Optimal solution value (if known)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate statistics
    constructive_avg = np.mean(constructive_results)
    vnd_avg = np.mean(vnd_results)
    
    # Set up box plot data
    data = [constructive_results, vnd_results]
    labels = ['Constructive', 'VND']
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Set colors
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='lightblue' if i == 0 else 'lightgreen')
    
    # Add scatter points for individual runs
    for i, results in enumerate([constructive_results, vnd_results]):
        x = np.random.normal(i + 1, 0.04, size=len(results))
        ax.scatter(x, results, alpha=0.6, s=30, c='blue' if i == 0 else 'green')
    
    # Add average lines
    ax.axhline(y=constructive_avg, color='blue', linestyle='dashed', alpha=0.7, label=f'Constructive Avg: {constructive_avg:.2f}')
    ax.axhline(y=vnd_avg, color='green', linestyle='dashed', alpha=0.7, label=f'VND Avg: {vnd_avg:.2f}')
    
    # Add optimal value line if provided
    if optimal_value > 0:
        ax.axhline(y=optimal_value, color='red', linestyle='dashed', linewidth=2, label=f'Optimal: {optimal_value:.2f}')
    
    # Set labels and title
    ax.set_ylabel('Solution Value (Penalty)')
    ax.set_title('Comparison of Solution Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
