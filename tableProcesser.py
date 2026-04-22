import numpy as np
import pandas as pd
import pickle
import sys, os
import itertools

sys.path.append(os.getcwd()+"/AggregatedPlottingScripts")
from tablesFromPkl import getTables

'''
Generic Caching utility for saving / loading long runtime function calls.
For any function F with parameters (x1,...,xn) where xi is a primative or vector of primatives, 
returns the cartesian product of arguments run through F, first searching cache for stored outputs of F.
Looks up cached results. Cache misses are computed and added to cache.
As it turns out, this is a built in python module functools. 
There is nothing new under the sun.
'''

class dataCache():
    def __init__(self, func, dataParams: dict, path=None):
        if path is None:
            self.path = f"{func.__name__}.pkl"
        else:
            self.path = path
        self.dataParams = dataParams
        self.func = func
        #FIXME will this make it more difficult to ensure it is always up to date when operating directly through main?
        try:
            with open(self.path, 'rb') as file:
                self.df = pickle.load(file)
                print("Loaded Cache from memory.")
        except FileNotFoundError:
            print("Cache not found. Generating from parameters...")
            columns = list(dataParams.keys()) + [func.__name__] #FIXME only give the function name if there is no dictionary output
            # print(columns)     
            self.df = pd.DataFrame(columns=columns)
            #FIXME call get_df. Making a blank one ensures get_df works the same.
            # self.get_df("a")
            self.quick_make()

    def quick_make(self):
        subdf = [] #each entry a dict with the parameters and results
        keys = self.dataParams.keys()
        cartesian = itertools.product(*[ self.dataParams[key] for key in keys ])
        for values in cartesian:
            subdf.append({key:value for (key,value) in zip(keys, values)} | {self.func.__name__:self.func(*values)})
        fresh_df = pd.DataFrame(subdf)
        with open(self.path, "wb") as file:
            pickle.dump(fresh_df,file)
        
        
def main():
    # Dataframe parameters are "species, particleEnergy, sample, betacut"
    species = ["Tau","Pion","Kaon"]
    particleEnergies = [10,100]
    samples = ["nominal"]
    # df = request_df(species,particleEnergies,samples,betaCuts)
    getTablesParams = {"species":species,"particleEnergies":particleEnergies,"samples":samples, "betaCut":[0.2]}
    tempCache = dataCache(getTables,getTablesParams)
    return

# columns =  [
# "species",
# "particleEnergy",
# "sample", #might need to expand this to physicsList and then sep by what variation is being applied
# "betaCut",

# "correct_noise",
# "incorrect_noise_as_signal",
# "incorrect_signal_as_noise",
# "correct_signal",
# "pred_noise",
# "pred_signal",
# "matched_truth_energy",
# "unmatched_truth_energy",
# "matched_pred_energy",
# "unmatched_pred_energy",
# "valid_events",
# "total_events",
# ]

if __name__=="__main__":
    main()