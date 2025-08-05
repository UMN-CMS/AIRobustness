import numpy as np
import pandas as pd
import pickle
import sys, os

sys.path.append(os.getcwd()+"/AggregatedPlottingScripts")
from tablesFromPkl import getTables

def make_dataframe():
    try:
        with open(".tableCache.pkl", 'rb') as file:
            df = pickle.load(file)
            print("path2")
    except FileNotFoundError:
        columns =  [
            "species",
            "particleEnergy",
            "sample",
            "betaCut",
            "correct_noise",
            "incorrect_noise_as_signal",
            "incorrect_signal_as_noise",
            "correct_signal",
            "pred_noise",
            "pred_signal",
            "matched_truth_energy",
            "unmatched_truth_energy",
            "matched_pred_energy",
            "unmatched_pred_energy",
            "valid_events",
            "total_events",
        ]
        df = pd.DataFrame(columns=columns)

    return df


def inspect_df(df, species, particleEnergy, sample, betaCut):
    
    for beta in betaCut:
        subdf = df[(df["species"]==species) & (df["particleEnergy"]==particleEnergy) & (df["sample"]==sample) & (df["betaCut"]==betaCut)] 
    if len(subdf) == 0:
        print(f"Generating {species} {particleEnergy} {sample} {betaCut}")
        return []
    else:
        return list(subdf.index)


def request_df(df, species, particleEnergy, sample, betaCut):
    if type(species) != list:
        species = [species] 
    if type(particleEnergy) != list:
        particleEnergy = [particleEnergy] 
    if type(sample) != list:
        sample = [sample] 
    if type(betaCut) != list:
        betaCut = [betaCut] 
    
    # index = inspect_df(df, species, particleEnergy, sample, betaCut)


    # if index == []:
    #     tempdf = {"species":species,"particleEnergy":particleEnergy,"sample":sample,"betaCut":betaCut}
    #     tempoutput = getTables(tempdf["species"],tempdf["particleEnergy"],tempdf["sample"],tempdf["betaCut"])
    #     for key in tempoutput:
    #         tempdf[key] = tempoutput[key]
    #     df = pd.concat([df, pd.DataFrame([tempdf])]).reset_index(drop=True)
    #     index = [len(df)]

    #     with open(".tableCache.pkl", 'wb') as file:
    #         pickle.dump(df, file)
    for samp in sample:
        for specie in species:
            for ptdEn in particleEnergy:
                print(f"Generating {specie} {ptdEn} {samp} {betaCut}")
                tempdf = getTables(specie,ptdEn,samp,betaCut)
                df = pd.concat([df, pd.DataFrame(tempdf)]).reset_index(drop=True)

    return df


df = make_dataframe()

species = ["Tau","Photon","Pion","Kaon"]
particleEnergies = ["e50"]

samples = [
    "nominal",
    # "BirkC1_0p006_25-02-04",
    # "EFTFP_1-25_25-02-04",
    # "EFTFP_20-40_EmaxBERT_20_20pi_25-02-04",
    # "EFTFP_3-15_25-02-04",
    # "EFTFP_3-35_25-02-04",
    # "EFTFP_5-25_25-02-04",
    # "EmaxBERT_3_pi6_25-02-04",
    # "EmaxBERT_9_18pi_25-02-04",
    # "EminQGSP_20_25-02-04",
    # "EminQGSP_6_25-02-04",
    # "FTFP_BERT_25-02-04",
    # "FTFP_BERT_EMM_25-02-04",
    # "FTFP_BERT_EMY_25-02-04",
    # "FTFP_BERT_EMZ_25-02-04",
    # "MELNRemoved",
    # "MELremoved",
    # "QGSP_FTFP_BERT_EML_25-02-04",
    # "zShift_1cm",
    # "removed1pct",
    # "removed10pct",
    ]

betaCuts = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]


df = request_df(df,species,particleEnergies,samples,betaCuts)

with open(".tableCache.pkl", 'wb') as file:
        pickle.dump(df, file)


"""

- Loads in the .tableCache if it exists, if not, make a generic version without any entries.
    Dont want to rewrite if it is not changed
    
- Request is made for a certain segment of the dataframe by available criteria.
- 
    
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

Considerations::
* Pass by value or reference for df to save on time. 
    Does python allow for by reference actually? Maybe a different language is key? Nah.
* 


"""

