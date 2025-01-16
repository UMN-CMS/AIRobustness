import numpy as np
import glob
from tqdm import tqdm
import pickle
import data_processing as dp


'''
Run this script for a given file path to convert the n number of pickles in the path 
into a single file that can be dealt with as one unit. This removes overhead of loading multiple files.
'''

# file_pattern = "/users/6/vadna042/hgcalml/hgcal_minimal_eval_example/output/singlePhotonOut9/*.pkl"
file_pattern = "/home/nstrobbe/shared/AIRobust/modifiedevents/singleTau24-11-25_E50_HENeighborsLayer8Removed/*.pkl"
file_limit = 1000
files = glob.glob(file_pattern)[:file_limit]
data, score_noise_filter, pass_noise_filter, out_gravnet = [], [], [], []

# extract data, each length is {file_limit}
for file in tqdm(files):
    # temp_load_data = np.load("/users/6/vadna042/airobustness/AIRobustness/Aggregated Plotting Scripts/NANOAOD_singlePhoton_50-50GeV_1_000_pos_ZMODIFIED3.npz", allow_pickle=True)
    temp_load_data = dp.load_data(file) #4 returns, all tensors
    data.append(temp_load_data[0])
    score_noise_filter.append(temp_load_data[1])
    pass_noise_filter.append(temp_load_data[2])
    out_gravnet.append(temp_load_data[3])

del files, file_pattern

sample = "TauE50Neighbors"

with open(f"pickles/{sample}/{sample}_data.pkl", 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickles/{sample}/{sample}_score_noise_filter.pkl", 'wb') as f:
    pickle.dump(score_noise_filter, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickles/{sample}/{sample}_pass_noise_filter.pkl", 'wb') as f:
    pickle.dump(pass_noise_filter, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickles/{sample}/{sample}_out_gravnet.pkl", 'wb') as f:
    pickle.dump(out_gravnet, f, protocol=pickle.HIGHEST_PROTOCOL)
