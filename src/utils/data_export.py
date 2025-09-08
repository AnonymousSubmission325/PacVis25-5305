# src/utils/data_export.py
import os
import pandas as pd
import ast
from src.utils.plotting import cut


def export_evaluation_results(results, data_names):
    """
    Exports the evaluation results to a CSV file in the 'src/results/records' directory,
    including the max_time parameter in the filename.

    Parameters:
    -----------
    results : list of dict
        A list of dictionaries containing evaluation metrics and other information.
    """
    # Ensure the directory for saving records exists
    output_dir = "src/results/records"
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename with max_time included
    output_filename = "evaluation_results_max_time_1000s.csv"
    results_path = os.path.join(output_dir, output_filename)


    # Convert the results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    print(f"Results exported")
    for dataset_name in data_names:
        print_results(results_df, results_path, dataset_name)
    # to_csv(results_path, index=False)
    
def remove_trailing_zeros(lst):
    # Remove trailing zeros from a list, but keep legitimate zeros inside
    if not lst:
        return lst
    while lst and lst[-1] == 0:
        lst = lst[:-1]
    return lst

def print_results(former_path, results_path, dataset_name):
    #print earlier results if datasets are not available
    df = pd.read_csv(results_path)
    df = df[df['Dataset'] == dataset_name]
    if len(df) == 0:
        print(f"Evaluation did not produce results.")
        return
    row = df.iloc[-1]

    technique_names = ast.literal_eval(row['technique_names'])
    stress_values = remove_trailing_zeros(ast.literal_eval(row['stress_values']))
    correlation_values = remove_trailing_zeros(ast.literal_eval(row['correlation_values']))
    silhouette_values = remove_trailing_zeros(ast.literal_eval(row['silhouette_values']))
    trust_values = remove_trailing_zeros(ast.literal_eval(row['trust_values']))
    avg_dist_values = remove_trailing_zeros(ast.literal_eval(row['avg_dist_values']))
    continuity_values = remove_trailing_zeros(ast.literal_eval(row['continuity_values']))
    neighborhood_hit_values = remove_trailing_zeros(ast.literal_eval(row['neighborhood_hit_values']))
    shepard_goodness_values = remove_trailing_zeros(ast.literal_eval(row['shepard_goodness_values']))
    distance_consistency_values = remove_trailing_zeros(ast.literal_eval(row['distance_consistency_values']))
    #cut to 3 decimal places
    print(f"Dataset: {dataset_name}")
    print(f"{'Metric':<25} | " + " | ".join(f"{name[:10]:>10}" for name in technique_names))
    print("-" * (25 + len(technique_names) * 13))
    print(f"{'Cosine - Stress':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in stress_values))
    print(f"{'Cosine - Correlation':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in correlation_values))
    print(f"{'Cosine - Silhouette':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in silhouette_values))
    print(f"{'Cosine - Trustworthiness':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in trust_values))
    print(f"{'Cosine - Avg. Distance':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in avg_dist_values))
    print(f"{'Continuity':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in continuity_values))
    print(f"{'Neighborhood Hit':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in neighborhood_hit_values))
    print(f"{'Shepard Goodness':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in shepard_goodness_values))
    print(f"{'Distance Consistency':<25} | " + " | ".join(f"{cut(val):10.2f}" for val in distance_consistency_values))

