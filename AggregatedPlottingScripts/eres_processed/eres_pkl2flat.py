import numpy as np
import pickle
import data_processing as dp
import glob
import sys

# Takes 2000 pickles from ~/hgcalml/hgcal_minimal_eval_example/output/<file>
# Processes down to the e_true_summed and e_pred_summed
# Saves to eres_processed/ with the name <Particle>_E<Energy>_<noiseEnergy>

#Data path and data containers

eResSample = sys.argv[1]
species, particleEnergy, noiseEnergy = eResSample.split("_")
sample = "eRes"
particleEnergy = int(particleEnergy[1:])

file_pattern = f"/users/6/vadna042/hgcalml/hgcal_minimal_eval_example/output/{eResSample}/*"
file_limit = 2000
files = glob.glob(file_pattern)[:file_limit]
print(len(files))
if len(files)<2000:
    print("Not enough files. Exiting.")
    exit()

true_cluster = np.empty(file_limit, dtype=object)
final_pred_hits = np.empty(file_limit, dtype=object)
energy_true = np.empty(file_limit, dtype=object)
energy_reco = np.empty(file_limit, dtype=object)

for i in range(len(files)):
    data, _, pass_noise_filter, out_gravnet = dp.load_data(files[i])
    true_cluster[i] = np.asarray(data.y)
    final_pred_hits[i] = np.asarray(dp.process_gravnet(pass_noise_filter, out_gravnet))
    energy_true[i] = data.x[:,0][true_cluster[i]==1]
    energy_reco[i] = data.x[:,0][final_pred_hits[i] > 0]

# layer_positions = np.loadtxt("unique_z.txt")

# true_cluster = np.array([ data[i].y for i in range(file_limit)], dtype=object) # real hits==1
# final_pred_hits = np.array([ dp.process_gravnet(pass_noise_filter[i], out_gravnet[i]) for i in range(file_limit)], dtype=object) #real clusterHits>0

# energy_true = np.array([ data[i].x[:,0][true_cluster[i]==1] for i in range(file_limit) ], dtype=object)
# energy_reco = np.array([ data[i].x[:,0][final_pred_hits[i] > 0] for i in range(file_limit) ], dtype=object)

# print(len(energy_true))

e_true_summed = np.asarray([np.sum(np.asarray(energy_true[i])) for i in range(file_limit)])
e_reco_summed = np.asarray([np.sum(np.asarray(energy_reco[i])) for i in range(file_limit)])

with open(f"{eResSample}.pkl",'wb') as file:
    pickle.dump(np.asarray([e_true_summed,e_reco_summed]),file)

