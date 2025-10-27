import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import hist
from tqdm import tqdm
import numpy as np
from itertools import product, combinations, combinations_with_replacement

import data_processing as dp
import plot_labels


def plot_ratio(data, metric, numerator, denominator, whichRatio):
    
    data1 = data[numerator[0]][f"{metric}_{numerator[1]}"]
    data2 = data[denominator[0]][f"{metric}_{denominator[1]}"]


    fig, axs = plt.subplots(nrows=2, figsize=(10,8), sharex=True, gridspec_kw={"hspace":0,"height_ratios":[3,1]})
    
    #who cares I just gotta force this plot out. If we need anymore this is gonna break so fast lol
    
    ratiobinning = [(0,22,22),
                    (22,30,4),
                    (0,70,35)]
    
    lower, upper, bins = ratiobinning[whichRatio]

    #Top plot
    histy1, histx1,_ = plt.hist(data1, bins=bins, range=[lower,upper])
    histy2, histx2,_ = plt.hist(data2, bins=bins, range = [lower,upper])

    #Top Plot Errorbars
    x_centers = np.asarray( [(x + histx1[i - 1])/2 for i, x in enumerate(histx1) if i > 0] )
    hist1err = np.sqrt(histy1)
    hist2err = np.sqrt(histy2)

    #Ratio plot
    ratiox = x_centers
    ratioy = np.asarray( [histy1[i]/histy2[i] if histy1[i] != 0 and histy2[i] != 0 else -1 for i in range(len(histy1))] ) #This is how its always meant to be
    #hist output and division is always >= 0 so use -1 as filter flag
    ratiox = ratiox[ratioy!=-1]
    hist1err = hist1err[ratioy!=-1]
    hist2err = hist2err[ratioy!=-1] 
    ratioy = ratioy[ratioy!=-1]
    ratioerr = (ratioy * np.sqrt(hist1err**-2 + hist2err**-2))
            
    return (ratiox, ratioy, ratioerr)

def rofR_helper(ratio1, ratio2):
    shared = list(set(ratio1[0]) & set(ratio2[0]))
    ratioratio = []
    ratioerrs = []
    for x in shared:
        i = np.where(ratio1[0] == x)[0]
        j = np.where(ratio2[0] == x)[0]
        ratioratio.append((ratio1[1][i]/ratio2[1][j])[0])
        ratioerrs.append((ratio1[1][i]/ratio2[1][j] * ((ratio1[2][i]/ratio1[1][i])**2 + (ratio2[2][j]/ratio2[1][j])**2)**0.5)[0])
    return np.asarray(shared), np.asarray(ratioratio), np.asarray(ratioerrs)

def get_fillBetween(sharedx, width, ratioerrs, ratioratio):
    fx = np.concatenate(np.asarray([ [sharedx[i] - (width/2), sharedx[i] + (width/2)] for i in range(len(sharedx))]))
    fupper = np.concatenate(np.asarray([ [x,x] for x in (ratioratio + ratioerrs) ])) 
    flower = np.concatenate(np.asarray([ [x,x] for x in (ratioratio - ratioerrs) ]))

    return fx, fupper,flower


def rofR2(ratio1, ratio2, ratio3, ratio4, ratio5, ratio6):
    sharedRad1, ratioratioRad1, ratioerrsRad1 = rofR_helper(ratio1,ratio2)
    sharedRad2, ratioratioRad2, ratioerrsRad2 = rofR_helper(ratio3,ratio4)
    sharedLon, ratioratioLon, ratioerrsLon = rofR_helper(ratio5,ratio6)

    widthRad1 = (sharedRad1[6]-sharedRad1[5]) # garbage: we cant rely on length since it isnt consistent
    widthRad2 = (sharedRad2[2]-sharedRad2[1]) #wow it keeps getting so much worse Lol
    widthLon = (sharedLon[6]-sharedLon[5])
    
    radF1, radFhigh1, radFlow1 = get_fillBetween(sharedRad1, widthRad1, ratioerrsRad1, ratioratioRad1)
    radF2, radFhigh2, radFlow2 = get_fillBetween(sharedRad2, widthRad2, ratioerrsRad2, ratioratioRad2)
    lonF, lonFhigh, lonFlow = get_fillBetween(sharedLon, widthLon, ratioerrsLon, ratioratioLon)


    fig, axs = plt.subplots(nrows=2, figsize=(10,8))
    fig.tight_layout(pad=4)
    axs[0].hlines(ratioratioRad1, sharedRad1-widthRad1/2, sharedRad1+widthRad1/2, color="black", zorder=3)
    axs[0].axhline(y=1, linestyle="--", linewidth="1", color="black")
    for i in range(len(ratioratioRad1)):
        axs[0].fill_between(radF1[2*i:2*i+2],radFhigh1[2*i:2*i+2],radFlow1[2*i:2*i+2], zorder=2, facecolor="grey", alpha=0.25)
    
    axs[0].hlines(ratioratioRad2, sharedRad2-widthRad2/2, sharedRad2+widthRad2/2, color="black", zorder=3)
    axs[0].axhline(y=1, linestyle="--", linewidth="1", color="black")
    for i in range(len(ratioratioRad2)):
        axs[0].fill_between(radF2[2*i:2*i+2],radFhigh2[2*i:2*i+2],radFlow2[2*i:2*i+2], zorder=2, facecolor="grey", alpha=0.25)
    
    axs[1].hlines(ratioratioLon, sharedLon-widthLon/2, sharedLon+widthLon/2, color="black", zorder=3)
    axs[1].axhline(y=1, linestyle="--", linewidth="1", color="black")
    for i in range(len(ratioratioLon)):
        axs[1].fill_between(lonF[2*i:2*i+2],lonFhigh[2*i:2*i+2],lonFlow[2*i:2*i+2], zorder=2, facecolor="grey", alpha=0.25)
    
    axs[0].set_xlabel(r"$\text{Radial Spread}\ E_{1\sigma} \ \ \text{[cm]}$", fontsize=14)
    axs[1].set_xlabel(r"$\text{Longitudinal Spread}\ E_{1\sigma} \ \ \text{[cm]}$", fontsize=14)
    axs[0].set_ylabel(r"$R$",fontsize=14)
    axs[1].set_ylabel(r"$R$",fontsize=14)

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].grid(which="minor", color="0.9")
    axs[1].grid(which="minor", color="0.9")
    axs[0].set_ylim(0.1,10)
    axs[1].set_ylim(0.1,10)
    axs[0].set_xlim(0,30)
    axs[1].set_xlim(0,70)
    axs[0].text(1,6,"QGSP_FTFP_BERT_EML vs. nominal", fontsize=10.5)
    axs[1].text(7/3,6,"QGSP_FTFP_BERT_EML vs. nominal", fontsize=10.5)
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[1].tick_params(axis='both', which='major', labelsize=11)
    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
    axs[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
    

    fig.savefig(f"rofR_Radial_Longitudinal_68.pdf")

    plt.close()


def main():
    #set event limit - max 1000 for our datasets
    file_limit = 1000

    #Z-positions. Precomputed
    layer_positions = np.loadtxt("unique_z.txt")

    metrics = [
            # "bestFit_r68", 
            "radial_68",
            "longitudinal_68",
            ]

    species = "Tau"
    particleEnergy = "e50"
    samples = ["nominal",["QGSP_FTFP_BERT_EML_25-02-04",]]
    labels = ["true", "pred"]
    
    results = {}
    ratios = {}

    if len(samples) == 0 or len(labels) == 0:
        print("What did you even expect to happen? Invalid input set.")
        return
    
    results[samples[0]] = dp.aggregate_data(samples[0], particleEnergy, species, layer_positions, file_limit, pred_cluster_cutoff=False)
    if len(samples) > 1:
        for sample in samples[1]:
            results[sample] = dp.aggregate_data(sample, particleEnergy, species, layer_positions, file_limit, pred_cluster_cutoff=False)


    ratios["histNomRad1"] = plot_ratio(results, "radial_68", ("nominal","true"), ("nominal","pred"), whichRatio=0)
    ratios["histNomRad2"] = plot_ratio(results, "radial_68", ("nominal","true"), ("nominal","pred"), whichRatio=1)
    ratios["histNomLon"] = plot_ratio(results, "longitudinal_68", ("nominal","true"), ("nominal","pred"), whichRatio=2)
    ratios["histEMLRad1"] = plot_ratio(results, "radial_68", ("QGSP_FTFP_BERT_EML_25-02-04","true"), ("QGSP_FTFP_BERT_EML_25-02-04","pred"), whichRatio=0)
    ratios["histEMLRad2"] = plot_ratio(results, "radial_68", ("QGSP_FTFP_BERT_EML_25-02-04","true"), ("QGSP_FTFP_BERT_EML_25-02-04","pred"), whichRatio=1)
    ratios["histEMLLon"] = plot_ratio(results, "longitudinal_68", ("QGSP_FTFP_BERT_EML_25-02-04","true"), ("QGSP_FTFP_BERT_EML_25-02-04","pred"), whichRatio=2)

    rofR2(ratios["histEMLRad1"],
           ratios["histNomRad1"],
           ratios["histEMLRad2"],
           ratios["histNomRad2"],
           ratios["histEMLLon"],
           ratios["histNomLon"])

    
if __name__ == '__main__':
    main()
