import pickle
import numpy as np
import torch
import glob
import pandas as pd
from tabulate import tabulate

import data_processing as dp

# Load set number of .pkl files
# file_pattern = r'C:\Users\tsoli\OneDrive\Documents\School\1 - University of Minnesota\Year 17\Year 1 Research\picklefiles\tau\*.pkl'

file_limit

# Clustering algorithm from plots3D.
def get_clustering(beta: np.array, X: np.array, threshold_beta: float = .2, threshold_dist: float = 0.5) -> np.array:
    n_points = beta.shape[0]
    select_condpoints = beta > threshold_beta
    indices_condpoints = np.nonzero(select_condpoints)[0]
    indices_condpoints = indices_condpoints[np.argsort(-beta[select_condpoints])]
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < threshold_dist]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < threshold_dist)]
    
    return clustering

# Parsing true input.
def process_data(data):
    true_energies = data.x[:, 0].numpy()
    true_clusters = data.y.numpy()
    return true_energies, true_clusters

# Parsing network output.
def process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet):
    beta = torch.sigmoid(out_gravnet[:, 0]).numpy()
    cluster_space_coords = out_gravnet[:, 1:].numpy()
    pred_clusters_pnf = get_clustering(beta, cluster_space_coords, threshold_beta=0.2, threshold_dist=0.5)
    pred_clusters = np.zeros_like(pass_noise_filter, dtype=np.int32)
    pred_clusters[pass_noise_filter] = pred_clusters_pnf
    return pred_clusters

def compute_statistics(true_energies, true_clusters, pred_clusters):
    total_energy = np.sum(true_energies)
    true_signal_energy = np.sum(true_energies[true_clusters != 0])
    true_noise_energy = np.sum(true_energies[true_clusters == 0])
    pred_signal_energy = np.sum(true_energies[pred_clusters != 0])
    pred_noise_energy = np.sum(true_energies[pred_clusters == 0])
    
    correct_noise = np.sum(true_energies[(true_clusters == 0) & (pred_clusters == 0)])
    incorrect_noise_as_signal = np.sum(true_energies[(true_clusters == 0) & (pred_clusters != 0)])
    correct_signal = np.sum(true_energies[(true_clusters != 0) & (pred_clusters != 0)])
    incorrect_signal_as_noise = np.sum(true_energies[(true_clusters != 0) & (pred_clusters == 0)])
    
    fractions = {
        'correct_noise': correct_noise / total_energy,
        'incorrect_noise_as_signal': incorrect_noise_as_signal / total_energy,
        'incorrect_signal_as_noise': incorrect_signal_as_noise / total_energy,
        'correct_signal': correct_signal / total_energy,
        'pred_noise': pred_noise_energy / total_energy,
        'pred_signal': pred_signal_energy / total_energy
    }
    
    return fractions

def generate_table_1(fractions):
    data = {
        'True / Predicted': ['True Noise', 'True Signal'],
        'Predicted Noise': [fractions['correct_noise'], fractions['incorrect_signal_as_noise']],
        'Predicted Signal': [fractions['incorrect_noise_as_signal'], fractions['correct_signal']]
    }
    df = pd.DataFrame(data)
    df.set_index('True / Predicted', inplace=True)
    print("Table 1:\n")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    return df

def compute_match_statistics(true_energies, true_clusters, pred_clusters):
    true_total_energy = np.sum(true_energies[true_clusters != 0])
    pred_total_energy = np.sum(true_energies[pred_clusters != 0])
    
    matched_truth_energy = np.sum(true_energies[(true_clusters != 0) & (pred_clusters != 0)])
    unmatched_truth_energy = np.sum(true_energies[(true_clusters != 0) & (pred_clusters == 0)])
    matched_pred_energy = np.sum(true_energies[(true_clusters != 0) & (pred_clusters != 0)])
    unmatched_pred_energy = np.sum(true_energies[(true_clusters == 0) & (pred_clusters != 0)])
    
    match_statistics = {
        'matched_truth_energy': matched_truth_energy / true_total_energy,
        'unmatched_truth_energy': unmatched_truth_energy / true_total_energy,
        'matched_pred_energy': matched_pred_energy / pred_total_energy,
        'unmatched_pred_energy': unmatched_pred_energy / pred_total_energy
    }
    
    return match_statistics

def generate_table_2(match_statistics):
    data = {
        'Matched/Unmatched': ['Matched Truth', 'Unmatched Truth', 'Matched Predicted', 'Unmatched Predicted'],
        'Energy Fraction': [
            match_statistics['matched_truth_energy'], 
            match_statistics['unmatched_truth_energy'],
            match_statistics['matched_pred_energy'], 
            match_statistics['unmatched_pred_energy']
        ]
    }
    df = pd.DataFrame(data)
    df.set_index('Matched/Unmatched', inplace=True)
    print("Table 2:\n")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    return df

def main():
    files = glob.glob(file_pattern)[:file_limit]
    all_fractions = []
    all_match_statistics = []

    sample = "nominal"
    particleEnergy = "e50"
    species = "Tau"

    all_data, all_pass_noise_filter, all_out_gravnet = dp.load_bulk_data(sample, particleEnergy, species)


    for file_path in files:
        data, score_noise_filter, pass_noise_filter, out_gravnet = load_data(file_path)
        true_energies, true_clusters = process_data(data)
        pred_clusters = process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet)
        
        fractions = compute_statistics(true_energies, true_clusters, pred_clusters)
        all_fractions.append(fractions)
        
        match_statistics = compute_match_statistics(true_energies, true_clusters, pred_clusters)
        all_match_statistics.append(match_statistics)
    
    average_fractions = {key: np.mean([d[key] for d in all_fractions]) for key in all_fractions[0]}
    average_match_statistics = {key: np.mean([d[key] for d in all_match_statistics]) for key in all_match_statistics[0]}
    
    print(f"Total events: {len(files)}")
    
    table_1 = generate_table_1(average_fractions)
    table_2 = generate_table_2(average_match_statistics)

if __name__ == '__main__':
    main()
