import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import altair as alt
import io
from datetime import datetime
import os

from algorithms.greedy import earliest_ready_time, least_penalty, combined_heuristic
from algorithms.vnd import vnd_algorithm
from algorithms.neighborhood import swap_flights, move_flight, swap_flights_between_runways
from utils.file_handler import parse_input_file, write_solution_file
from utils.visualization import plot_runway_schedule, plot_comparative_results
from utils.metrics import calculate_solution_value, calculate_gap

# Set page configuration
st.set_page_config(
    page_title="Airport Runway Scheduling Optimizer",
    page_icon="✈️",
    layout="wide",
)

# Title and description
st.title("✈️ Otimizador de Agendamento de Pistas de Aeroporto")
st.markdown("""
Esta aplicação ajuda a otimizar o agendamento de pousos e decolagens em um aeroporto com múltiplas pistas.
O objetivo é minimizar as penalidades por atraso, respeitando as restrições de segurança entre voos consecutivos.
""")

# Sidebar for configuration
st.sidebar.header("Configuração")

# File upload
uploaded_file = st.sidebar.file_uploader("Carregar arquivo de instância", type=["txt"])

# Only proceed if a file is uploaded
if uploaded_file is not None:
    # Parse input file
    try:
        num_flights, num_runways, release_times, processing_times, penalties, waiting_times = parse_input_file(uploaded_file)
        
        st.sidebar.success(f"Instância carregada com {num_flights} voos e {num_runways} pistas")
        
        # Display instance details in expandable section
        with st.expander("Detalhes da Instância"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Voos")
                flight_data = pd.DataFrame({
                    "Voo": [f"Voo {i+1}" for i in range(num_flights)],
                    "Tempo de Liberação": release_times,
                    "Tempo de Processamento": processing_times,
                    "Penalidade por Unidade de Atraso": penalties
                })
                st.dataframe(flight_data)
            
            with col2:
                st.subheader("Matriz de Tempos de Espera")
                waiting_df = pd.DataFrame(waiting_times)
                waiting_df.index = [f"Voo {i+1}" for i in range(num_flights)]
                waiting_df.columns = [f"Voo {i+1}" for i in range(num_flights)]
                st.dataframe(waiting_df)
        
        # Algorithm selection
        st.sidebar.header("Seleção de Algoritmos")
        constructive_algorithm = st.sidebar.selectbox(
            "Algoritmo Construtivo",
            ["Earliest Ready Time", "Least Penalty", "Combined Heuristic"]
        )
        
        use_vnd = st.sidebar.checkbox("Aplicar VND (Variable Neighborhood Descent)", value=True)
        
        # VND configuration if selected
        if use_vnd:
            st.sidebar.subheader("Configuração do VND")
            max_iterations = st.sidebar.slider("Número Máximo de Iterações", 10, 1000, 100)
            neighborhood_order = st.sidebar.multiselect(
                "Ordem das Vizinhanças",
                ["Swap Flights", "Move Flight", "Swap Between Runways"],
                ["Swap Flights", "Move Flight", "Swap Between Runways"]
            )
        
        # Reference optimal solution
        st.sidebar.header("Optimal Solution (if known)")
        optimal_value = st.sidebar.number_input("Optimal Solution Value", min_value=0.0, value=0.0)
        
        # Number of runs for statistical analysis
        st.sidebar.header("Statistical Analysis")
        num_runs = st.sidebar.slider("Number of Runs", 1, 50, 10)
        
        # Run optimization
        if st.sidebar.button("Run Optimization"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Storage for results
            constructive_results = []
            vnd_results = []
            constructive_times = []
            vnd_times = []
            
            # Run multiple times for statistical analysis
            for run in range(num_runs):
                status_text.text(f"Running iteration {run+1}/{num_runs}...")
                progress_bar.progress((run+1) / num_runs)
                
                # Select and run constructive algorithm
                start_time = time.time()
                if constructive_algorithm == "Earliest Ready Time":
                    initial_solution = earliest_ready_time(release_times, processing_times, waiting_times, num_runways)
                elif constructive_algorithm == "Least Penalty":
                    initial_solution = least_penalty(release_times, processing_times, waiting_times, penalties, num_runways)
                else:  # Combined Heuristic
                    initial_solution = combined_heuristic(release_times, processing_times, waiting_times, penalties, num_runways)
                
                constructive_time = time.time() - start_time
                constructive_value = calculate_solution_value(initial_solution, release_times, processing_times, waiting_times, penalties)
                constructive_results.append(constructive_value)
                constructive_times.append(constructive_time)
                
                # Run VND if selected
                if use_vnd:
                    start_time = time.time()
                    
                    # Map selected neighborhoods to functions
                    neighborhood_functions = []
                    if "Swap Flights" in neighborhood_order:
                        neighborhood_functions.append(swap_flights)
                    if "Move Flight" in neighborhood_order:
                        neighborhood_functions.append(move_flight)
                    if "Swap Between Runways" in neighborhood_order:
                        neighborhood_functions.append(swap_flights_between_runways)
                    
                    improved_solution = vnd_algorithm(
                        initial_solution,
                        release_times,
                        processing_times,
                        waiting_times,
                        penalties,
                        neighborhood_functions,
                        max_iterations
                    )
                    
                    vnd_time = time.time() - start_time
                    vnd_value = calculate_solution_value(improved_solution, release_times, processing_times, waiting_times, penalties)
                    vnd_results.append(vnd_value)
                    vnd_times.append(vnd_time)
                
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Calculate statistics
            constructive_avg = np.mean(constructive_results)
            constructive_best = min(constructive_results)
            constructive_avg_time = np.mean(constructive_times)
            
            # Display results
            st.header("Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Constructive Algorithm Results")
                st.write(f"Algorithm: {constructive_algorithm}")
                st.write(f"Average Solution Value: {constructive_avg:.2f}")
                st.write(f"Best Solution Value: {constructive_best:.2f}")
                st.write(f"Average Execution Time: {constructive_avg_time:.4f} seconds")
                
                if optimal_value > 0:
                    constructive_gap = calculate_gap(constructive_best, optimal_value)
                    st.write(f"GAP to Optimal: {constructive_gap:.2f}%")
                
                # Get the best constructive solution for visualization
                if constructive_algorithm == "Earliest Ready Time":
                    best_constructive = earliest_ready_time(release_times, processing_times, waiting_times, num_runways)
                elif constructive_algorithm == "Least Penalty":
                    best_constructive = least_penalty(release_times, processing_times, waiting_times, penalties, num_runways)
                else:  # Combined Heuristic
                    best_constructive = combined_heuristic(release_times, processing_times, waiting_times, penalties, num_runways)
                
                st.write("Best Constructive Solution:")
                fig = plot_runway_schedule(best_constructive, release_times, processing_times, penalties, num_runways, waiting_times)
                st.pyplot(fig)
            
            if use_vnd:
                with col2:
                    st.subheader("VND Results")
                    vnd_avg = np.mean(vnd_results)
                    vnd_best = min(vnd_results)
                    vnd_avg_time = np.mean(vnd_times)
                    
                    st.write(f"Average Solution Value: {vnd_avg:.2f}")
                    st.write(f"Best Solution Value: {vnd_best:.2f}")
                    st.write(f"Average Execution Time: {vnd_avg_time:.4f} seconds")
                    
                    if optimal_value > 0:
                        vnd_gap = calculate_gap(vnd_best, optimal_value)
                        st.write(f"GAP to Optimal: {vnd_gap:.2f}%")
                    
                    # Get the best VND solution for visualization
                    if constructive_algorithm == "Earliest Ready Time":
                        initial = earliest_ready_time(release_times, processing_times, waiting_times, num_runways)
                    elif constructive_algorithm == "Least Penalty":
                        initial = least_penalty(release_times, processing_times, waiting_times, penalties, num_runways)
                    else:  # Combined Heuristic
                        initial = combined_heuristic(release_times, processing_times, waiting_times, penalties, num_runways)
                    
                    neighborhood_functions = []
                    if "Swap Flights" in neighborhood_order:
                        neighborhood_functions.append(swap_flights)
                    if "Move Flight" in neighborhood_order:
                        neighborhood_functions.append(move_flight)
                    if "Swap Between Runways" in neighborhood_order:
                        neighborhood_functions.append(swap_flights_between_runways)
                    
                    best_vnd = vnd_algorithm(
                        initial,
                        release_times,
                        processing_times,
                        waiting_times,
                        penalties,
                        neighborhood_functions,
                        max_iterations
                    )
                    
                    st.write("Best VND Solution:")
                    fig = plot_runway_schedule(best_vnd, release_times, processing_times, penalties, num_runways, waiting_times)
                    st.pyplot(fig)
            
            # Comparative analysis
            st.header("Comparative Analysis")
            
            if use_vnd:
                # Create a comparison chart between constructive and VND
                fig = plot_comparative_results(constructive_results, vnd_results, optimal_value)
                st.pyplot(fig)
                
                # Results table for all runs
                results_df = pd.DataFrame({
                    "Run": range(1, num_runs + 1),
                    f"{constructive_algorithm} Value": constructive_results,
                    f"{constructive_algorithm} Time (s)": constructive_times,
                    "VND Value": vnd_results,
                    "VND Time (s)": vnd_times
                })
                st.dataframe(results_df)
            else:
                # Create a comparison chart for just constructive
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(constructive_results, bins=10, alpha=0.7)
                ax.axvline(constructive_avg, color='red', linestyle='dashed', linewidth=1, label=f'Average: {constructive_avg:.2f}')
                if optimal_value > 0:
                    ax.axvline(optimal_value, color='green', linestyle='dashed', linewidth=1, label=f'Optimal: {optimal_value:.2f}')
                ax.set_xlabel('Solution Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {constructive_algorithm} Solution Values')
                ax.legend()
                st.pyplot(fig)
                
                # Results table for all runs
                results_df = pd.DataFrame({
                    "Run": range(1, num_runs + 1),
                    f"{constructive_algorithm} Value": constructive_results,
                    f"{constructive_algorithm} Time (s)": constructive_times
                })
                st.dataframe(results_df)
            
            # Export best solution
            st.header("Export Solution")
            
            # Select the best solution to export
            if use_vnd:
                final_solution = best_vnd
                final_value = vnd_best
            else:
                final_solution = best_constructive
                final_value = constructive_best
            
            solution_buffer = io.StringIO()
            write_solution_file(solution_buffer, final_solution, final_value)
            solution_str = solution_buffer.getvalue()
            
            st.text_area("Solution Output", solution_str, height=200)
            
            # Create download button
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.download_button(
                label="Download Solution",
                data=solution_str,
                file_name=f"solution_{timestamp}.txt",
                mime="text/plain"
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Display instructions when no file is uploaded
    st.info("Please upload an instance file to begin optimization.")
    
    with st.expander("File Format Instructions"):
        st.markdown("""
        ### Expected File Format
        ```
        1 number_of_flights
        2 number_of_runways
        3
        4 array r (release times)
        5 array c (processing times)
        6 array p (penalties)
        7
        8 matrix t (waiting times)
        ```
        
        ### Example
        ```
        1 6
        2 2
        3
        4 5 25 15 40 75 50
        5 15 25 20 30 15 25
        6 55 90 61 120 45 50
        7
        8 0 10 15 8 21 15
        9 10 0 10 13 15 20
        10 17 9 0 10 14 8
        11 11 13 12 0 10 10
        12 5 10 15 20 0 12
        13 5 10 15 20 28 0
        ```
        """)
    
    # Display example visualization
    with st.expander("Example Visualization"):
        st.write("Exemplo de visualização de escalonamento de voos em múltiplas pistas será mostrado aqui após o carregamento dos dados.")

# Footer
st.markdown("---")
st.markdown("Airport Runway Scheduling Optimizer | Developed with Streamlit")
