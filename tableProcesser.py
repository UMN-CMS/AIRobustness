import numpy as np
import pandas as pd
import pickle
import sys, os
import itertools

# sys.path.append(os.getcwd()+"/AggregatedPlottingScripts")
# from tablesFromPkl import getTables


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


    def make_df(self):
        # find all missing elements
        # run the function and create the complementary_df
        # Merge self.df with complementary_df
        # Save self.df to self.path
        return

    def get_df(self,a): #Returns the segment of df that we want. Assumes its already in there
        print(a)
        # if request not in df:
            # make_df
        # df = self.df[request] 

        return #the segment of df that is requested. 
    
    def whatami(self):
        print("WhatamI? ",self.df)
        return
        
        
# #Incorporate
# def inspect_df(df, species, particleEnergy, sample, betaCut):
    
#     for beta in betaCut:
#         subdf = df[(df["species"]==species) & (df["particleEnergy"]==particleEnergy) & (df["sample"]==sample) & (df["betaCut"]==betaCut)] 
#     if len(subdf) == 0:
#         print(f"Generating {species} {particleEnergy} {sample} {betaCut}")
#         return []
#     else:
#         return list(subdf.index)

###############################################

def foo(sample,species,energy): # always returns an int
    if species=="crow":
        spec = 1
    elif species=="cow":
        spec = 2
    return len(sample) * spec * energy 

def main():
    #short sweet tester to make sure that everything works on a generic example
    #function foo(param1=,param2=,...)
    #becomes dataCacher.get_datacache("filename",foo,{param1:,param2:})
    params = {
        "sample":["nominal"],
        "species":["cow","crow"],
        "energy":[10,30,50]
    }
    params2 = {
        "sample":["baba","nominal"],
        "species":["cow","crow"],
        "energy":[10,20,50]
    }
    dataCacher = dataCache(foo,params,"cow.pkl")
    dataCacher.whatami()
    return


if __name__=="__main__":
    main()

""" 
- Request is made for a certain segment of the dataframe by available criteria.
    How to solve the problem of initial state without having a huge comparison at the end? does it matter? It always matters but only in the end, never the beginning. 
        The beginning defines the spawning of ideas and forms. Critters walking out into harsh sunlight and dry rocks
        As time weathers them, adaptations evolve their state until a hardened warrior emerges.
        This is never an end state, but it diffuses to its oppressor which has alwasy been time and space.
        The computer is no different. Except a critter does not live when not attended by an architect and can stop evolving when its function has run out.
        In this way, the beast has no life, but can be brought to life as an extension of the agentic. Therefore, the machine is not living, but a piece of armor for the softened being that seeks shelter.
- Request any of sample, species, ptdEn, or beta as lists or single values.
    Returns df containing everything that matches the description.
        process by index since the indexing will only ever increase and not change the existing ones
        Start by asking for each instance what is required.
        If found then give index, if not, give the n+1 index and generate the product.
        Do we add it here or what?
- If something is missing, it will try to acquire it for you before generating the output df. 
    Future version will save time by batching all beta variants together to avoid loading every time.

"""


#junk for pushing values quick
def main2():
    #Dataframe parameters are "species, particleEnergy, sample, betacut"
    # species = ["Tau","Pion","Kaon"]
    # particleEnergies = ["e10","e100"]
    # samples = ["nominal"]
    # df = request_df(species,particleEnergies,samples,betaCuts)

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