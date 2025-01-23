import pickle
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def load_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        score_noise_filter = pickle.load(f)
        pass_noise_filter = pickle.load(f)
        out_gravnet = pickle.load(f)
    return data, score_noise_filter, pass_noise_filter, out_gravnet

def load_data_bulk(sample):
    with open(f"pickles/{sample}/{sample}_data.pkl", 'rb') as f:
        data = pickle.load(f)
    with open(f"pickles/{sample}/{sample}_score_noise_filter.pkl", 'rb') as f:
        score_noise_filter = pickle.load(f)
    with open(f"pickles/{sample}/{sample}_pass_noise_filter.pkl", 'rb') as f:
        pass_noise_filter = pickle.load(f)
    with open(f"pickles/{sample}/{sample}_out_gravnet.pkl", 'rb') as f:
        out_gravent = pickle.load(f)
    
    return data, score_noise_filter, pass_noise_filter, out_gravent

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

# Collecting raw data.
def process_data_matching_hist(data):
    '''
    Returns desired data entries from the data column from an event pickle file.
    Returns used by energy resolution matching functions primarily.
    '''
    true_energies = data.x[:, 0].numpy()
    true_clusters = data.y.numpy()
    eta_values = data.x[:, 1].numpy()
    true_pdgids = data.truth_cluster_props[:, 4].numpy()
    return true_energies, true_clusters, eta_values, true_pdgids

def process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet):
    """Process the network output to predict clusters."""
    sigmoid = lambda x : (1+np.exp(-x)) ** (-1)
    beta = np.array(sigmoid(out_gravnet[:, 0]))
    cluster_space_coords = out_gravnet[:, 1:].numpy()
    pred_clusters_pnf = get_clustering(beta, cluster_space_coords, threshold_beta=0.2, threshold_dist=0.5)
    pred_clusters = np.zeros_like(pass_noise_filter, dtype=np.int32)
    pred_clusters[pass_noise_filter] = pred_clusters_pnf
    
    unique, counts = np.unique(pred_clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    final_pred_hits = np.array([cluster if cluster_counts[cluster] >= 100 else -2 for cluster in pred_clusters])
    
    return final_pred_hits

def process_gravnet_matching_hist(score_noise_filter, pass_noise_filter, out_gravnet):
    sigmoid = lambda x : (1+np.exp(-x)) ** (-1)
    beta = np.array(sigmoid(out_gravnet[:, 0]))
    cluster_space_coords = out_gravnet[:, 1:].numpy()
    pred_clusters_pnf = get_clustering(beta, cluster_space_coords, threshold_beta=0.2, threshold_dist=0.5)
    pred_clusters = np.zeros_like(pass_noise_filter, dtype=np.int32)
    pred_clusters[pass_noise_filter] = pred_clusters_pnf
    return pred_clusters

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
    nearest_layer_com = np.argmin(np.abs(layer_positions - z_com))
    # nearest_layer_com = layer_positions[nearest_layer_com] #old method, we want the layer index instead

    distances = np.abs(zpos[cluster_indices] - z_com)
    sorted_indices = np.argsort(distances)
    sorted_energies = energies[cluster_indices][sorted_indices]
    cumulative_energies = np.cumsum(sorted_energies)
    total_energy = cumulative_energies[-1]
    
    longitudinal_68 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.68 * total_energy)]
    longitudinal_95 = distances[sorted_indices][np.searchsorted(cumulative_energies, 0.95 * total_energy)]
    
    return nearest_layer_com, longitudinal_68, longitudinal_95

def find_first_hit(z_vals,layer_positions):
    firstZ = np.round(sorted(z_vals)[0])
    firstLayer = np.argwhere(firstZ==layer_positions)[0][0]
    return firstLayer

def maxELayer(z, energy, layer_positions):
    z = np.round(z)
    energy_per_layer = [ np.sum(energy[z==lz]) for lz in layer_positions]
    return np.argmax(energy_per_layer)


def pca(x,y,z, energy = None):
    '''
    Calculates the First principle component using PCA for a 3d data set.
    This acts as a 3d line of best fit. Uses sklearn.decompossition.pca (FAST!!!)
    Inputs: x,y,z,energy(optional) must all be 1d arrays of the same length
    Returns: Mean point of x,y,z and unit vector of the first principle component.
    '''
    if energy == None:
        data = np.array([x,y,z]).T
    else: 
        data = np.array([x,y,z,energy]).T
    
    pca = PCA(n_components=1)
    pca.fit(data)
    datamean = np.mean(data, axis=0)
    line = pca.components_
    return datamean[:3], line[0][:3] 

def pcaTrace(x,y,z,energy=None, color="black"):
    '''
    Returns a plotly.go trace of a best fit line for a 3d data set.
    Inputs: x,y,z,energy 1d arrays of equal length
    '''
    datamean, line = pca(x,y,z,energy)
    linepoints = line * np.mgrid[-30:30:2j][:, np.newaxis] + datamean

    trace = go.Scatter3d(
        x = linepoints[:,0],
        y = linepoints[:,1],
        z = linepoints[:,2],
        line=dict(color=color, width=2)
        )
    return trace

def point_line_dist(p,a,vec):
    '''
    Returns (closest) distance from a point 'p' to a line
    with unit direction 'vec' that passes through point 'a'.
    Alternative calculation scheme using the dot product instead of the cross product.
    '''
    p = np.array(p)
    return np.linalg.norm((a-p) - np.outer(np.dot((a-p),vec),vec), axis=1)

def e_radius(distances, energy, e_limit):
    '''
    Given a best fit line of energized hits, returns the radius of a cylinder that encapsulates e_limit percent of the data.
    Inputs: Distances: 1d array of absolute distances per hit from fit line
            energy:  1d array of energy for the hits
            e_limit: total energy cutoff to include within the cylinder from [0,1]
    '''
    if e_limit > 1 or e_limit < 0:
        print("e_limit must be float|int from [0,1]")
        return 0
    distances = np.array(distances) #Type check type check
    energy = np.array(energy)
    indexsort = np.argsort(distances)
    distances = distances[indexsort]
    energy = energy[indexsort]
    e_total = np.sum(energy)
    e_running = 0
    for i in range(len(distances)):
        e_running += energy[i]
        if e_running >= e_total * e_limit:
            return distances[i] 
    print("You shouldn't have gotten here")
    return distances[-1]

def calculate_chi2(absolute_dists, energy_true):
    return sum((absolute_dists**2) / (len(energy_true) - 4))

def calculate_absolute_distances(x,y,z,energy=None):
    '''
    Calculates the absolute distance from a 3d best fit line using PCA
    '''
    point, line = pca(x, y, z, energy)
    pees = np.array([x, y, z]).T
    return point_line_dist(pees, point, line)


# Match algorithm from matching.py
def match(clustering1, clustering2, weights=None, threshold=0.2, noise_index=0):
    if weights is None:
        weights = np.ones_like(clustering1)
    cluster_ids1, cluster_indices1 = np.unique(clustering1, return_inverse=True)
    cluster_ids2, cluster_indices2 = np.unique(clustering2, return_inverse=True)
    n_clusters1 = cluster_ids1.shape[0]
    n_clusters2 = cluster_ids2.shape[0]
    
    # Pre-calculate all 'areas' for all clusters
    areas1 = {id: weights[clustering1 == id].sum() for id in cluster_ids1}
    areas2 = {id: weights[clustering2 == id].sum() for id in cluster_ids2}
    
    # Make list of all pairs
    a = np.repeat(np.arange(n_clusters1), n_clusters2)
    b = np.repeat(np.expand_dims(np.arange(n_clusters2), -1), n_clusters1, axis=1).T.ravel()
    pairs = np.vstack((cluster_ids1[a], cluster_ids2[b])).T
    
    if noise_index is not None:
        # Remove pairs with a noise index in there
        pairs = pairs[~np.amax(pairs == noise_index, axis=-1).astype(bool)]
    
    # Calculate weighted ioms
    ioms = np.zeros(pairs.shape[0])
    intersections = np.zeros(pairs.shape[0])
    for i_pair, (id1, id2) in enumerate(pairs):
        intersection = weights[(clustering1 == id1) & (clustering2 == id2)].sum()
        minimum = min(areas1[id1], areas2[id2])
        ioms[i_pair] = intersection / minimum
        intersections[i_pair] = intersection
    
    # Sort
    order = np.argsort(intersections)[::-1]
    intersections = intersections[order]
    ioms = ioms[order]
    pairs = pairs[order]
    
    # Matching algo
    canhavemorematches_1 = set(cluster_ids1)
    canhavemorematches_2 = set(cluster_ids2)
    matched_1 = set()
    matched_2 = set()
    matches = []
    
    for iom, intersection, (i1, i2) in zip(ioms, intersections, pairs):
        if iom < threshold:
            continue
        if i1 not in canhavemorematches_1 or i2 not in canhavemorematches_2:
            continue
        if i1 in matched_1 and i2 in matched_2:
            continue
        # Make the match
        matches.append([i1, i2, iom])
        if i1 in matched_1:
            i2s = [j2 for j1, j2, _ in matches if j1 == i1]
            canhavemorematches_2.difference_update(i2s)
        elif i2 in matched_2:
            i1s = [j1 for j1, j2, _ in matches if j2 == i2]
            canhavemorematches_1.difference_update(i1s)
        matched_1.add(i1)
        matched_2.add(i2)
    
    if len(matches) == 0:
        print('Warning: No matches at all')
        return [], [], []
    
    matches = np.array(matches)
    i1s, i2s, ioms = matches[:, 0].astype(np.int32), matches[:, 1].astype(np.int32), matches[:, 2]
    return i1s, i2s, ioms

# Group matching
def group_matching(i1s, i2s, return_lists=True):
    match_dict_1_to_2 = {}
    match_dict_2_to_1 = {}
    is_used_1 = set()
    is_used_2 = set()
    for i1, i2 in zip(i1s, i2s):
        i1 = int(i1)
        i2 = int(i2)
        i1_used = i1 in is_used_1
        i2_used = i2 in is_used_2
        if not(i1_used) and not(i2_used):
            match_dict_1_to_2[i1] = [i2]
            match_dict_2_to_1[i2] = [i1]
            is_used_1.add(i1)
            is_used_2.add(i2)
        elif i1_used and i2_used:
            raise Exception(
                f'Detected many-to-many match:'
                f' [left]{i1} and [right]{i2} are both already matched to something else'
            )
        elif i1 in is_used_1:
            match_dict_1_to_2[i1].append(i2)
            match_dict_2_to_1.pop(i2, None)
        elif i2 in is_used_2:
            match_dict_2_to_1[i2].append(i1)
            match_dict_1_to_2.pop(i1, None)
    if return_lists:
        matches = [[[k], v] for k, v in match_dict_1_to_2.items()]
        matches.extend([[v, [k]] for k, v in match_dict_2_to_1.items() if len(v) > 1])
    else:    
        matches = [[k, (v if len(v) > 1 else v[0])] for k, v in match_dict_1_to_2.items()]
        matches.extend([[v, k] for k, v in match_dict_2_to_1.items() if len(v) > 1])
    return matches

# Make matches
def make_matches(event, prediction, tbeta=0.2, td=0.5, clustering=None):
    if clustering is None:
        clustering = cluster(prediction, tbeta, td)
    i1s, i2s, _ = match(event.y, clustering, weights=event.energy)
    matches = group_matching(i1s, i2s)
    return matches

def get_category(truth_ids):
    em_ids = np.array([11, 22, 111])
    mip_ids = np.array([13])
    if np.all(np.isin(truth_ids, em_ids)):
        return 0  # EM
    elif np.all(np.isin(truth_ids, mip_ids)):
        return 2  # MIP
    elif np.any(np.isin(truth_ids, em_ids)) or np.any(np.isin(truth_ids, mip_ids)):
        return 3  # MIX
    else:
        return 1  # HAD


# Histogram generator.
def accumulate_histograms(hist_data, data, score_noise_filter, pass_noise_filter, out_gravnet):
    true_energies, true_clusters, eta_values, true_pdgids = process_data_matching_hist(data)
    pred_clusters = process_gravnet_matching_hist(score_noise_filter, pass_noise_filter, out_gravnet)
    
    i1s, i2s, _ = match(true_clusters, pred_clusters, weights=true_energies)
    matches = group_matching(i1s, i2s)
    
    for truth_ids, pred_ids in matches:
        if 0 in truth_ids or -1 in pred_ids:
            continue
        
        mask_truth = np.isin(true_clusters, truth_ids)
        mask_pred = np.isin(pred_clusters, pred_ids)
        
        true_energy_sum = true_energies[mask_truth].sum()
        pred_energy_sum = true_energies[mask_pred].sum()
        
        ratio = pred_energy_sum / true_energy_sum
        eta_region = np.abs(eta_values[mask_truth].mean()) < 2.1
        
        category = get_category(np.unique(true_pdgids[mask_truth]))
        labels = ['EM', 'HAD', 'MIP', 'MIX']
        
        if eta_region:
            hist_data['low_eta'][labels[category]].append(ratio)
        else:
            hist_data['high_eta'][labels[category]].append(ratio)

def fullmask(data):
    '''
    The masking functions are meant to be called as a final step on the data.
    These will simulate different failure modes of the detector as represented in changes in data.
    '''
    return np.ones_like(data, dtype=bool)