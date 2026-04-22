import numpy as np

import data_processing as dp

# Load set number of .pkl files

file_limit = 1000

'''
Computes Table 1 (Signal or noise correctly or incorrectly identified)
Computes Table 2 (match/unmatched truth/prediction energy)
Outputs table values in dictionary
Not all datasamples have valid predictions - depends on the network. If this occurs, table 2 is reduced.
'''

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
    print(f"========table 1========")
    print(f"{fractions['correct_noise']} {fractions['incorrect_noise_as_signal']}")
    print(f"{fractions['incorrect_signal_as_noise']} {fractions['correct_signal']}")
    return

def compute_match_statistics(true_energies, true_clusters, pred_clusters):
    true_signal_energy = np.sum(true_energies[true_clusters > 0])
    pred_signal_energy = np.sum(true_energies[pred_clusters > 0])
    
    matched_truth_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters > 0)]) / true_signal_energy
    unmatched_truth_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters <= 0)]) / true_signal_energy
    
    if pred_signal_energy == 0:
        match_statistics = {
            'matched_truth_energy': matched_truth_energy,
            'unmatched_truth_energy': unmatched_truth_energy,
        }
        return match_statistics
    
    matched_pred_energy = np.sum(true_energies[(true_clusters > 0) & (pred_clusters > 0)]) / pred_signal_energy
    unmatched_pred_energy = np.sum(true_energies[(true_clusters <= 0) & (pred_clusters > 0)]) / pred_signal_energy
    
    match_statistics = {
        'matched_truth_energy': matched_truth_energy,
        'unmatched_truth_energy': unmatched_truth_energy,
        'matched_pred_energy': matched_pred_energy,
        'unmatched_pred_energy': unmatched_pred_energy,
    }

    return match_statistics

def generate_table_2(match_statistics):
    print(f"========table 2========")
    print(f"{match_statistics['matched_truth_energy']} {match_statistics['unmatched_truth_energy']}")
    print(f"{match_statistics['matched_pred_energy']} {match_statistics['unmatched_pred_energy']}")
    return

# def main():
def getTables(species,particleEnergy,sample,betaCut):
    if type(betaCut) != list:
        betaCut = [betaCut]
    
    #Species, particleEnergy,sample will not be lists. betaCut might.

    outputList = []
    all_data, all_pass_noise_filter, all_out_gravnet = dp.load_data_bulk(sample, particleEnergy,species)
    for beta in betaCut:
        all_fractions = {
            'correct_noise': [],
            'incorrect_noise_as_signal': [],
            'incorrect_signal_as_noise': [],
            'correct_signal': [],
            'pred_noise': [],
            'pred_signal': [],    
        }
        all_match_statistics = {
            'matched_truth_energy': [],
            'unmatched_truth_energy': [],
            'matched_pred_energy': [],
            'unmatched_pred_energy': [],
        }
        
        for i in range(file_limit):
            data = all_data[i]
            pass_noise_filter = all_pass_noise_filter[i]
            out_gravnet = all_out_gravnet[i]
            
            true_energies, true_clusters = process_data(data)
            pred_clusters = dp.process_gravnet(pass_noise_filter, out_gravnet, cutoff = False, tbeta = beta)
            
            fractions = compute_statistics(true_energies, true_clusters, pred_clusters)
            for key in fractions:
                all_fractions[key].append(fractions[key])
            
            match_statistics = compute_match_statistics(true_energies, true_clusters, pred_clusters)
            for key in match_statistics:
                all_match_statistics[key].append(match_statistics[key])
        

        average_fractions = {key: np.mean(all_fractions[key]) for key in all_fractions}
        average_match_statistics = {key: np.mean(all_match_statistics[key]) for key in all_match_statistics}
        print(f"Total events: {len(all_match_statistics['matched_pred_energy'])}/{file_limit}")
        
        # table_1 = generate_table_1(average_fractions)
        # table_2 = generate_table_2(average_match_statistics)
        output = {"species":species,"particleEnergy":particleEnergy,"sample":sample,"betaCut":beta}
        for key in average_fractions:
            output[key] = average_fractions[key]
        for key in average_match_statistics:
            output[key] = average_match_statistics[key]
        output["valid_events"] = len(all_match_statistics['matched_pred_energy'])
        output["total_events"] = file_limit
        outputList.append(output)

    return outputList


# if __name__ == '__main__':
#     main()

# print(getTables("Photon","e50","FTFP_BERT_EMM_25-02-04"))