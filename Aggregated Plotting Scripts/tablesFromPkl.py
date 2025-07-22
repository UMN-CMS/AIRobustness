import pickle
import numpy as np
import torch
import glob
import pandas as pd
from tabulate import tabulate

import data_processing as dp

# Load set number of .pkl files
# file_pattern = r'C:\Users\tsoli\OneDrive\Documents\School\1 - University of Minnesota\Year 17\Year 1 Research\picklefiles\tau\*.pkl'

file_limit = 1000

# Parsing true input.
def process_data(data):
    true_energies = data.x[:, 0].numpy()
    true_clusters = data.y.numpy()
    return true_energies, true_clusters

def compute_statistics(true_energies, true_clusters, pred_clusters):
    total_energy = np.sum(true_energies)
    pred_signal_energy = np.sum(true_energies[pred_clusters > 0])
    pred_noise_energy = np.sum(true_energies[pred_clusters <= 0])
    
    correct_noise = np.sum(true_energies[(true_clusters <= 0) & (pred_clusters <= 0)])
    incorrect_noise_as_signal = np.sum(true_energies[(true_clusters <= 0) & (pred_clusters > 0)])
    correct_signal = np.sum(true_energies[(true_clusters > 0) & (pred_clusters > 0)])
    incorrect_signal_as_noise = np.sum(true_energies[(true_clusters > 0) & (pred_clusters <= 0)])
    
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
    # data = {
    #     'True / Predicted': ['True Noise', 'True Signal'],
    #     'Predicted Noise': [fractions['correct_noise'], fractions['incorrect_signal_as_noise']],
    #     'Predicted Signal': [fractions['incorrect_noise_as_signal'], fractions['correct_signal']]
    # }
    # df = pd.DataFrame(data)
    # df.set_index('True / Predicted', inplace=True)
    # print("Table 1:\n")
    # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    # The above is better for terminal inspection
    # The below is better for converting to the google sheet
    print(f"========table 1========")
    print(f"{fractions['correct_noise']} {fractions['incorrect_noise_as_signal']}")
    print(f"{fractions['incorrect_signal_as_noise']} {fractions['correct_signal']}")
    return

def compute_match_statistics(true_energies, true_clusters, pred_clusters):
    true_signal_energy = np.sum(true_energies[true_clusters > 0])
    pred_signal_energy = np.sum(true_energies[pred_clusters > 0])
    
    matched_truth_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters > 0)])
    unmatched_truth_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters <= 0)])
    matched_pred_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters > 0)])
    unmatched_pred_energy = np.sum(true_energies[(true_clusters <= 0) & (pred_clusters > 0)])
    
    match_statistics = {
        'matched_truth_energy': matched_truth_energy / true_signal_energy,
        'unmatched_truth_energy': unmatched_truth_energy / true_signal_energy,
        'matched_pred_energy': matched_pred_energy / pred_signal_energy,
        'unmatched_pred_energy': unmatched_pred_energy / pred_signal_energy
    }
    
    return match_statistics

def generate_table_2(match_statistics):
    # data = {
    #     'Matched/Unmatched': ['Matched Truth', 'Unmatched Truth', 'Matched Predicted', 'Unmatched Predicted'],
    #     'Energy Fraction': [
    #         match_statistics['matched_truth_energy'], 
    #         match_statistics['unmatched_truth_energy'],
    #         match_statistics['matched_pred_energy'], 
    #         match_statistics['unmatched_pred_energy']
    #     ]
    # }
    # df = pd.DataFrame(data)
    # df.set_index('Matched/Unmatched', inplace=True)
    # print("Table 2:\n")
    # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
    print(f"========table 2========")
    print(f"{match_statistics['matched_truth_energy']} {match_statistics['unmatched_truth_energy']}")
    print(f"{match_statistics['matched_pred_energy']} {match_statistics['unmatched_pred_energy']}")
    return

def main():
    all_fractions = []
    all_match_statistics = []
    valid_events = 0
    sample = "FTFP_BERT_EMM_25-02-04"
    particleEnergy = "e50"
    species = "Kaon"

    all_data, all_pass_noise_filter, all_out_gravnet = dp.load_data_bulk(sample, particleEnergy, species)

    for i in range(file_limit):
        data = all_data[i]
        pass_noise_filter = all_pass_noise_filter[i]
        out_gravnet = all_out_gravnet[i]
        
        

        true_energies, true_clusters = process_data(data)
        pred_clusters = dp.process_gravnet(pass_noise_filter, out_gravnet, cutoff = False, tbeta = 0.9)
        
        print(np.unique(pred_clusters))

        
        if not np.any(pred_clusters > 0):
            continue
        
        valid_events += 1
        fractions = compute_statistics(true_energies, true_clusters, pred_clusters)
        all_fractions.append(fractions)
        
        match_statistics = compute_match_statistics(true_energies, true_clusters, pred_clusters)
        all_match_statistics.append(match_statistics)
    
    average_fractions = {key: np.mean([d[key] for d in all_fractions]) for key in all_fractions[0]}
    average_match_statistics = {key: np.mean([d[key] for d in all_match_statistics]) for key in all_match_statistics[0]}
    
    print(f"Total events: {valid_events}/{file_limit}")
    
    table_1 = generate_table_1(average_fractions)
    table_2 = generate_table_2(average_match_statistics)

if __name__ == '__main__':
    main()
