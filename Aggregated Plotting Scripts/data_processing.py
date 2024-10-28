import pickle
import numpy as np
import torch

def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        score_noise_filter = pickle.load(f)
        pass_noise_filter = pickle.load(f)
        out_gravnet = pickle.load(f)
    return data, score_noise_filter, pass_noise_filter, out_gravnet

def get_clustering(beta, X, threshold_beta=0.2, threshold_dist=0.5):
    """Cluster points based on beta values and distances."""
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

def process_data(data):
    """Extract relevant information from the data."""
    true_energies = data.x[:, 0].numpy()
    true_clusters = data.y.numpy()
    xpos = data.x[:, 5].numpy()
    ypos = data.x[:, 6].numpy()
    zpos = data.x[:, 7].numpy()
    return true_energies, true_clusters, xpos, ypos, zpos

def process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet):
    """Process the network output to predict clusters."""
    beta = torch.sigmoid(out_gravnet[:, 0]).numpy()
    cluster_space_coords = out_gravnet[:, 1:].numpy()
    pred_clusters_pnf = get_clustering(beta, cluster_space_coords, threshold_beta=0.2, threshold_dist=0.5)
    pred_clusters = np.zeros_like(pass_noise_filter, dtype=np.int32)
    pred_clusters[pass_noise_filter] = pred_clusters_pnf
    
    unique, counts = np.unique(pred_clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    final_pred_hits = np.array([cluster if cluster_counts[cluster] >= 100 else -2 for cluster in pred_clusters])
    
    return final_pred_hits

def calculate_radial_shower_spread(cluster_indices, xpos, ypos, energies):
    """Calculate the Radial Shower Spread (68% and 95% energy radii)."""
    x_com = np.sum(xpos[cluster_indices] * energies[cluster_indices]) / np.sum(energies[cluster_indices])
    y_com = np.sum(ypos[cluster_indices] * energies[cluster_indices]) / np.sum(energies[cluster_indices])
    
    distances = np.sqrt((xpos[cluster_indices] - x_com)**2 + (ypos[cluster_indices] - y_com)**2)
    sorted_indices = np.argsort(distances)
    sorted_energies = energies[cluster_indices][sorted_indices]
    cumulative_energies = np.cumsum(sorted_energies)
    total_energy = cumulative_energies[-1]
    
    radial_68 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.68 * total_energy)]
    radial_95 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.95 * total_energy)]
    
    return radial_68, radial_95

def calculate_longitudinal_shower_spread(cluster_indices, zpos, energies, layer_positions):
    """Calculate the Longitudinal Shower Spread (68% and 95% energy lengths)."""
    z_com = np.sum(zpos[cluster_indices] * energies[cluster_indices]) / np.sum(energies[cluster_indices])
    nearest_layer_com = layer_positions[np.argmin(np.abs(layer_positions - z_com))]
    
    distances = np.abs(zpos[cluster_indices] - z_com)
    sorted_indices = np.argsort(distances)
    sorted_energies = energies[cluster_indices][sorted_indices]
    cumulative_energies = np.cumsum(sorted_energies)
    total_energy = cumulative_energies[-1]
    
    longitudinal_68 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.68 * total_energy)]
    longitudinal_95 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.95 * total_energy)]
    
    return nearest_layer_com, longitudinal_68, longitudinal_95
