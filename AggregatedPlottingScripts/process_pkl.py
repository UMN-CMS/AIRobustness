import glob
from tqdm import tqdm
import pickle
import data_processing as dp
import os

'''
Converts N events into the data, score_noise_filter, pass_noise_filter, out_gravnet files each containing N entries.
Helps to improve loading times in data_plotting.py  
Make sure there is a directory called pickles, update file_pattern and sample, then run
'''
samples = [x.split("/")[-1] for x in glob.glob("/home/nstrobbe/mahon/hgcalmlSingularity/hgcal_minimal_eval_example/output/*25-02-04")]

species = ["Tau","Kaon","Pion","Photon"]

for specie in species:
    for sample in samples:

        file_pattern = f"/home/nstrobbe/mahon/hgcalmlSingularity/hgcal_minimal_eval_example/output/{sample}/NANOAOD_single{specie}*.pkl"
        
        file_limit = 1000
        files = glob.glob(file_pattern)[:file_limit]
        data, score_noise_filter, pass_noise_filter, out_gravnet = [], [], [], []

        for file in tqdm(files):
            # temp_load_data = np.load("/users/6/vadna042/airobustness/AIRobustness/AggregatedPlottingScripts/NANOAOD_singlePhoton_50-50GeV_1_000_pos_ZMODIFIED3.npz", allow_pickle=True)
            temp_load_data = dp.load_data(file) #4 returns, all tensors
            data.append(temp_load_data[0])
            score_noise_filter.append(temp_load_data[1])
            pass_noise_filter.append(temp_load_data[2])
            out_gravnet.append(temp_load_data[3])

        del files, file, file_pattern, file_limit

        if not os.path.isdir(f"pickles/{specie}/e50/{sample}"):
            os.makedirs(f"pickles/{specie}/e50/{sample}")

        print(f"copying to {specie}/e50/{sample}")
        with open(f"pickles/{specie}/e50/{sample}/data.pkl", 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"pickles/{specie}/e50/{sample}/score_noise_filter.pkl", 'wb') as f:
            pickle.dump(score_noise_filter, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"pickles/{specie}/e50/{sample}/pass_noise_filter.pkl", 'wb') as f:
            pickle.dump(pass_noise_filter, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"pickles/{specie}/e50/{sample}/out_gravnet.pkl", 'wb') as f:
            pickle.dump(out_gravnet, f, protocol=pickle.HIGHEST_PROTOCOL)

        del data, score_noise_filter, pass_noise_filter, out_gravnet, temp_load_data

        