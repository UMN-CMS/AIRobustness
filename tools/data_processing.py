import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from skspatial.objects import Points, Line

# Clustering algorithm from plots3D.
def get_clustering(beta, X, threshold_beta=0.2, threshold_dist=0.5):
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

# Parsing network output.
def process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet):
    sigmoid = lambda x : (1+np.exp(-x)) ** (-1)
    beta = np.array(sigmoid(out_gravnet[:, 0]))
    cluster_space_coords = out_gravnet[:, 1:].numpy()
    pred_clusters_pnf = get_clustering(beta, cluster_space_coords, threshold_beta=0.2, threshold_dist=0.5)
    pred_clusters = np.zeros_like(pass_noise_filter, dtype=np.int32)
    pred_clusters[pass_noise_filter] = pred_clusters_pnf
    
    # Count hits per cluster
    unique, counts = np.unique(pred_clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Create final_pred_hits with clusters having less than 100 hits labeled as -2
    final_pred_hits = np.array([cluster if cluster_counts[cluster] >= 100 else -2 for cluster in pred_clusters])
    
    return final_pred_hits

def pcadepreciated3(x,y,z,energy=None):
    '''
    Calculates the First principle component using PCA for a 3d data set.
    This acts as a 3d line of best fit. Uses np.linalg.svd (SLOW!!!)
    Inputs: x,y,z,energy must all be 1d arrays of the same length
    Returns: Mean point of x,y,z and unit vector of the first principle component.
    '''
    if energy==None:
        data = np.array((x,y,z)).T
    else: 
        data = np.array((x,y,z,energy)).T

    datamean = data.mean(axis=0)
    _, _, line = np.linalg.svd(data - datamean)

    return datamean[:3], line[0][:3]

def pcadepreciated2(x,y,z, energy=None):
    '''
    Calculates the First principle component using PCA for a 3d data set.
    This acts as a 3d line of best fit. Uses skspatial points,line (SLOW!!!)
    Inputs: x,y,z,energy must all be 1d arrays of the same length
    Returns: Mean point of x,y,z and unit vector of the first principle component.
    '''
    if energy == None:
        data = Points(np.array((x,y,z)).T)
    else:
        data = Points(np.array((x,y,z,energy)).T)
    fitLine = Line.best_fit(data)
    return np.array(fitLine.point)[:3], np.array(fitLine.direction)[:3]
  
def pca(x,y,z, energy = None):
    '''
    Calculates the First principle component using PCA for a 3d data set.
    This acts as a 3d line of best fit. Uses sklearn.decompossition.pca (FAST!!!)
    Inputs: x,y,z,energy must all be 1d arrays of the same length
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
    distances = np.array(distances) #Type checks
    energy = np.array(energy)
    indexsort = np.argsort(distances)
    distances = distances[indexsort] 
    energy = energy[indexsort]
    e_total = np.sum(energy)
    e_running = 0
    for i in range(len(distances)):
        if e_running < e_total * e_limit:
            e_running += energy[i]
        else:
            return distances[i] 