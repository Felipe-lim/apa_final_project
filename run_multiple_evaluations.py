import subprocess
import os
import io
import csv
import pandas as pd
import argparse
import numpy as np
import sys

def parse_evaluation_output(output):
  csv_marker = "Final Results (CSV Format - Excluding LB=0 Instances):"
  try:
    csv_start_index = output.index(csv_marker) + len(csv_marker)
    csv_data_str = output[csv_start_index:].strip()
    if not csv_data_str:
      return []
    csv_file = io.StringIO(csv_data_str)
    reader = csv.DictReader(csv_file)
    results = []
    for row in reader:
      processed_row = {}
      for key, value in row.items():
        clean_key = key.strip()
        processed_row[clean_key] = value

      for key in ['Penalty', 'Time', 'Ref_Value', 'GAP', 'Ref_Type']:
        try:
          val = processed_row.get(key)
          if key in ['Penalty', 'Time', 'Ref_Value', 'GAP']:
            if val is not None and val.strip() != '':
              processed_row[key] = float(val)
            else:
              processed_row[key] = np.nan
          elif key == 'Ref_Type':
            processed_row[key] = val if val and val.strip() else None
          else:
             processed_row[key] = val
        except (ValueError, TypeError):
          processed_row[key] = np.nan
          print(f"Aviso: Não foi possível converter valor para '{key}'. Definindo como NaN.")
        except KeyError:
          processed_row[key] = np.nan
          print(f"Aviso: Chave esperada '{key}' não encontrada na linha. Definindo como NaN.")
      results.append(processed_row)
    return results
  except ValueError:
    print(f"Aviso: Marcador CSV '{csv_marker}' não encontrado na saída do script.")
    return []
  except Exception as e:
    print(f"ERRO ao parsear dados CSV: {e}")
    return []

def run_evaluation_script(script_path):
  try:
    script_dir = os.path.dirname(script_path) if os.path.dirname(script_path) else "."
    python_executable = sys.executable
    result = subprocess.run(
        [python_executable, os.path.basename(script_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=script_dir,
        encoding='utf-8',
        errors='replace'
    )
    if result.returncode != 0:
        print(f"ERRO: Script de avaliação encerrou com código {result.returncode}.")
        return None
    return result.stdout
  except FileNotFoundError:
    print(f"ERRO: Script não encontrado em {script_path}")
    return None
  except Exception as e:
    print(f"ERRO ao executar script {script_path}: {e}")
    return None

def main(num_runs, output_dir):
  script_to_run = "evaluate_test_instances.py"
  script_path = os.path.abspath(script_to_run)
  all_results = []

  if not os.path.exists(script_path):
      print(f"ERRO: O script '{script_to_run}' não foi encontrado em '{script_path}'.")
      return

  if not os.path.exists(output_dir):
      try:
        os.makedirs(output_dir)
      except OSError as e:
        print(f"ERRO ao criar o diretório de saída '{output_dir}': {e}")
        return

  print(f"Executando {script_to_run} {num_runs} vezes...")

  run_success_count = 0
  for i in range(num_runs):
    print(f"  Execução {i+1}/{num_runs}...")
    output = run_evaluation_script(script_path)
    if output:
      parsed_data = parse_evaluation_output(output)
      if parsed_data:
          for row in parsed_data:
              row['Run'] = i + 1
          all_results.extend(parsed_data)
          run_success_count += 1
      else:
          print(f"  Aviso: Falha ao parsear a saída da execução {i+1}.")
    else:
        print(f"  Aviso: Falha ao executar a execução {i+1}.")

  if run_success_count < num_runs:
      print(f"\nAviso: {num_runs - run_success_count} de {num_runs} execuções falharam ou não puderam ser parseadas.")

  if not all_results:
    print("Nenhum resultado válido coletado. Encerrando.")
    return

  try:
    df = pd.DataFrame(all_results)

    required_cols = ['Instance', 'Algorithm', 'Penalty', 'Time', 'GAP', 'Ref_Type']
    if not all(col in df.columns for col in required_cols):
        print(f"ERRO: Colunas essenciais ({required_cols}) ausentes nos dados coletados.")
        return

    ref_type_map = df.drop_duplicates(subset=['Instance'])[['Instance', 'Ref_Type']].set_index('Instance')['Ref_Type']

    for col in ['Penalty', 'Time', 'GAP']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    agg_funcs = {
        'Penalty': [
            ('BestPenalty', lambda x: np.nanmin(x) if not x.isnull().all() else np.nan),
            ('WorstPenalty', lambda x: np.nanmax(x) if not x.isnull().all() else np.nan),
            ('MeanPenalty', lambda x: np.nanmean(x) if not x.isnull().all() else np.nan)
        ],
        'Time': [('MeanTime', lambda x: np.nanmean(x) if not x.isnull().all() else np.nan)],
        'GAP': [
            ('BestGAP', lambda x: np.nanmin(x) if not x.isnull().all() else np.nan),
            ('MeanGAP', lambda x: np.nanmean(x) if not x.isnull().all() else np.nan)
        ]
    }

    aggregated_df = df.groupby(['Instance', 'Algorithm']).agg(agg_funcs)
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
    aggregated_df = aggregated_df.reset_index()

    pivoted_df = aggregated_df.pivot_table(
        index='Instance',
        columns='Algorithm',
        values=['Penalty_BestPenalty', 'Penalty_WorstPenalty', 'Penalty_MeanPenalty',
                'Time_MeanTime', 'GAP_BestGAP', 'GAP_MeanGAP']
    )
    pivoted_df.columns = [f"{col[1]}_{col[0]}" for col in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()

    if 'Instance' in pivoted_df.columns:
      pivoted_df['Ref_Type'] = pivoted_df['Instance'].map(ref_type_map)

    algorithms_order = ['GRASP', 'Greedy', 'ILS', 'VND']
    metrics_order = ['Penalty_BestPenalty', 'Penalty_WorstPenalty', 'Penalty_MeanPenalty',
                     'Time_MeanTime', 'GAP_BestGAP', 'GAP_MeanGAP']

    final_columns = ['Instance']
    if 'Ref_Type' in pivoted_df.columns:
      final_columns.append('Ref_Type')

    present_algorithms = df['Algorithm'].unique()
    for algo in algorithms_order:
        if algo in present_algorithms:
            for metric in metrics_order:
                col_name = f"{algo}_{metric}"
                if col_name in pivoted_df.columns:
                    final_columns.append(col_name)
    
    # Ensure all columns from pivot are included, maintaining the order of final_columns
    ordered_cols = final_columns + [col for col in pivoted_df.columns if col not in final_columns]
    pivoted_df = pivoted_df[ordered_cols]

  except Exception as e:
      print(f"ERRO durante a agregação dos dados: {e}")
      return

  if not pivoted_df.empty:
    output_filename = os.path.join(output_dir, "resultados_agregados.csv")
    try:
        pivoted_df.to_csv(output_filename, index=False, float_format='%.4f')
        print(f"\nArquivo de resumo salvo com sucesso: {output_filename}")
    except Exception as e:
        print(f"\nERRO ao salvar o arquivo de resumo {output_filename}: {e}")
  else:
    print("\nNenhum resultado agregado para salvar.")

  print("\nScript finalizado.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Executa evaluate_test_instances.py multiplas vezes e agrega os resultados.")
  parser.add_argument("-n", "--num_runs", type=int, default=10, help="Número de vezes para executar o script de avaliação (padrão: 10).")
  parser.add_argument("-o", "--output_dir", type=str, default="aggregated_results", help="Diretório para salvar os resultados agregados em CSV (padrão: aggregated_results).")
  args = parser.parse_args()

  main(args.num_runs, args.output_dir) 
