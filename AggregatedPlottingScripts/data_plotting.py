import matplotlib.pyplot as plt
import hist
from tqdm import tqdm
import numpy as np
from itertools import product, combinations, combinations_with_replacement

import data_processing as dp
import plot_labels

'''
Mass produces Ratio plots and Ratio of Ratio plots for valid combinations of metrics, samples, species.
Plots collected into pdfs using mergePDF.sh. 
File type for merged PDF determined by plot_ratio() and rofR() functions. 
PNG will show resolution loss in pdf form. 
'''

def plot_energy_resolution(results, sample):
    hist_data = results["hist_data"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    etas = []
    for label in ['EM', 'HAD', 'MIP', 'MIX']:
        etas.append(hist_data["low_eta"][label])
        etas.append(hist_data["high_eta"][label])

    upperBound = max([max(x) for x in etas if x != []])
    lowerBound = min([min(x) for x in etas if x != []])
        
    for label in ['EM', 'HAD', 'MIP', 'MIX']:
        low_eta_data = hist_data['low_eta'][label]
        high_eta_data = hist_data['high_eta'][label]
        ax1.hist(low_eta_data, bins=np.linspace(lowerBound, upperBound, 50), alpha=0.5, label=label, histtype='step', density=True, linewidth=2)
        ax2.hist(high_eta_data, bins=np.linspace(lowerBound, upperBound, 50), alpha=0.5, label=label, histtype='step', density=True, linewidth=2)
    
    ax1.set_title(f'{sample} |η| < 2.1')
    ax2.set_title(f'{sample} |η| > 2.1')
    for ax in (ax1, ax2):
        ax.set_xlabel('E(pred) / E(true)')
        ax.set_ylabel('Arbitrary Units')
        ax.legend()

    
    plt.savefig(f"zenergy_resolution_{sample}.png") # I just want this is appear last in the document

def plot_ratio(data, metric, numerator, denominator, species, sig=None, pred_cluster_cutoff=True):
    '''
    Generic histogram ratio plotting utility.
    hist1 and hist2 must be 1d arrays
    formatText correlates variable names/values with plot format text. 
    '''    
    
    data1 = data[numerator[0]][f"{metric}_{numerator[1]}"]
    data2 = data[denominator[0]][f"{metric}_{denominator[1]}"]
    formatText = plot_labels.plot_labels_select(metric, numerator, denominator, species, pred_cluster_cutoff)

    # binning
    avg = np.mean([np.mean(data1), np.mean(data2)]) # Plots center of both hists
    std = np.mean([np.std(data1),np.std(data2)])

    if metric == "firstLayer" or metric == "maxELayer" or metric == "coe_layers": 
        sig = None

    # histy1, histx1 = np.histogram(data1, bins = formatText["bins"], range = [formatText["lower"], formatText["upper"]])
    # histy2, histx2 = np.histogram(data2, bins = formatText["bins"], range = [formatText["lower"], formatText["upper"]])

    fig, axs = plt.subplots(nrows=2, figsize=(10,8), sharex=True, gridspec_kw={"hspace":0,"height_ratios":[3,1]})
    
    # axs[0].bar(histx1[1:], histy1, width=(formatText["upper"]-formatText["lower"])/formatText["bins"], 
    #             yerr = hist1err, align = "edge", color="blue", ecolor="blue", fill = False,
    #             label = f"{formatText['label1']}, $\mu$ {np.mean(data1):.2f}, $\sigma$ {np.std(data1):.2f}")
    # axs[0].bar(histx2[1:], histy2, width=(formatText["upper"]-formatText["lower"])/formatText["bins"], 
    #             yerr = hist2err, align = "edge", color="orange", ecolor="orange", fill = False,
    #             label = f"{formatText['label2']}, $\mu$ {np.mean(data2):.2f}, $\sigma$ {np.std(data2):.2f}\n{len(data1)} events plotted")

    if sig==0:
        upper = np.max(np.concatenate((data1,data2))) + 0.5*std
        lower = np.min(np.concatenate((data1,data2))) - 0.5*std
        bins  = 100
    elif sig == None:
        upper = formatText["upper"]
        lower = formatText["lower"]
        bins  = formatText["bins"]
    else:
        upper = avg + sig*std
        lower = avg - sig*std
        bins  = 100


    #Top plot
    histy1, histx1,_ = axs[0].hist(data1, bins = bins, histtype="step",
                                range = [lower,upper], alpha=0.80,
                                label = f"{formatText['label1']}, $\mu$ {np.mean(data1):.2f}, $\sigma$ {np.std(data1):.2f}",
                                color="blue")
    histy2, histx2,_ = axs[0].hist(data2, bins = bins, histtype="step",
                                range = [lower,upper], alpha=0.80,
                                label = f"{formatText['label2']}, $\mu$ {np.mean(data2):.2f}, $\sigma$ {np.std(data2):.2f}\n{len(data1)} events plotted",
                                color="orange")

    #Top Plot Errorbars
    x_centers = np.asarray( [(x + histx1[i - 1])/2 for i, x in enumerate(histx1) if i > 0] )
    hist1err = np.sqrt(histy1)
    hist2err = np.sqrt(histy2)

    axs[0].errorbar(x_centers, histy1, hist1err, color="blue", ls="none")
    axs[0].errorbar(x_centers, histy2, hist2err, color="orange", ls="none")

    #Ratio plot
    ratiox = x_centers
    ratioy = np.asarray( [histy1[i]/histy2[i] if histy1[i] != 0 and histy2[i] != 0 else -1 for i in range(len(histy1))] ) #This is how its always meant to be
    #hist output and division is always >= 0 so use -1 as filter flag
    ratiox = ratiox[ratioy!=-1]
    hist1err = hist1err[ratioy!=-1]
    hist2err = hist2err[ratioy!=-1] 
    ratioy = ratioy[ratioy!=-1]
    ratioerr = (ratioy * np.sqrt(hist1err**-2 + hist2err**-2))


    axs[1].scatter(ratiox,ratioy, color="black")
    axs[1].errorbar(ratiox,ratioy, ratioerr, color="black", ls="none")
    axs[1].axhline(y=1, linestyle="--", linewidth="1", color="black")


    #Format
    axs[0].set_title(formatText["title"])
    axs[1].set_xlabel(formatText["x_axis"])
    axs[0].set_ylabel("Count")
    axs[1].set_ylabel("Ratio")
    axs[0].legend()
    axs[1].set_ylim(0,2.4)

    fig.savefig(formatText["saveas"])
    # plt.show()
    plt.close()
            
    return (ratiox, ratioy, ratioerr)

def rofR(ratio1, ratio2, metric, ratioName1, ratioName2, species, pred_cluster_cutoff=True):
    ratio1x = ratio1[0]
    ratio1y = ratio1[1]
    ratio1err = ratio1[2]
    ratio2x = ratio2[0]
    ratio2y = ratio2[1]
    ratio2err = ratio2[2]

    fig, axs = plt.subplots(nrows=3, figsize=(10,8), sharex=True, gridspec_kw={"hspace":0})
    formatText = plot_labels.plot_labels_select(metric, (ratioName1,"true"), (ratioName1,"pred"), species, pred_cluster_cutoff)

    axs[0].scatter(ratio1x,ratio1y, color="black",
                   label=f"{ratioName1} true/pred")
    axs[0].errorbar(ratio1x, ratio1y, ratio1err, color="black", ls="none")
    axs[0].axhline(y=1, linestyle="--", linewidth="1", color="black")
    axs[1].scatter(ratio2x,ratio2y, color="black",
                   label=f"{ratioName2} true/pred")
    axs[1].errorbar(ratio2x, ratio2y, ratio2err, color="black", ls="none")
    axs[1].axhline(y=1, linestyle="--", linewidth="1", color="black")

    #ratio of ratios section
    shared = list(set(ratio1x) & set(ratio2x))
    ratioratio = []
    ratioerrs = []
    for x in shared:
        i = np.where(ratio1x == x)[0]
        j = np.where(ratio2x == x)[0]
        ratioratio.append((ratio1y[i]/ratio2y[j])[0])
        ratioerrs.append((ratio1y[i]/ratio2y[j] * ((ratio1err[i]/ratio1y[i])**2 + (ratio2err[j]/ratio2y[j])**2)**0.5)[0])

    axs[2].scatter(shared,ratioratio, color="black")
    axs[2].errorbar(shared,ratioratio,ratioerrs, color="black", ls="none")
    axs[2].axhline(y=1, linestyle="--", linewidth="1", color="black")

    axs[0].set_title(f"Ratio of Ratios: {formatText['title']}")
    axs[2].set_xlabel(formatText["x_axis"])
    axs[0].set_ylabel("Ratio1")
    axs[1].set_ylabel("Ratio2")
    axs[2].set_ylabel("Ratio1/Ratio2")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_ylim(0,2.4)
    axs[1].set_ylim(0,2.4)
    axs[2].set_ylim(0,5)
    
    fig.savefig(f"rofR_{metric}_{ratioName1}_{ratioName2}.png")

    plt.close()

def main():
    #set event limit - max 1000 for our datasets
    file_limit = 1000

    #Z-positions. Precomputed
    layer_positions = np.loadtxt("unique_z.txt")

    metrics = [
            "bestFit_r99",
            "bestFit_r95", 
            "bestFit_r68",
            "bestFit_r99-95",
            "bestFit_r95-68", 
            "radial_68",
            "radial_95",
            "coe_layers",
            "longitudinal_68",
            "longitudinal_95",
            "chi2",
            "abs_dists",
            "avg_weighted_dist",
            "firstLayer", 
            "maxELayer",
            "emRatio",
            ]

    # Samples follows the given structure
    # [sampleBase, [compSample,...]:optional]
    # SampleBase is compared against all given compSamples.
    # Minimum example is just sampleBase.
    # No direct comparisons between compSamples, only sampleBase (base dataset generally)

    species = "Tau"
    particleEnergy = "e50"
    samples = ["nominal",[
                        #   "BirkC1_0p006_25-02-04",
                        #   "EFTFP_1-25_25-02-04",
                        #   "EFTFP_20-40_EmaxBERT_20_20pi_25-02-04",
                        #   "EFTFP_3-15_25-02-04",
                        #   "EFTFP_3-35_25-02-04",
                        #   "EFTFP_5-25_25-02-04",
                        #   "EmaxBERT_3_pi6_25-02-04",
                        #   "EmaxBERT_9_18pi_25-02-04",
                        #   "EminQGSP_20_25-02-04",
                        #   "EminQGSP_6_25-02-04",
                        #   "FTFP_BERT_25-02-04",
                        #   "FTFP_BERT_EMM_25-02-04",
                        #   "FTFP_BERT_EMY_25-02-04",
                        #   "FTFP_BERT_EMZ_25-02-04",
                        #   "MELNRemoved",
                        #   "MELremoved",
                        #   "QGSP_FTFP_BERT_EML_25-02-04",
                        #   "zShift_1cm",
                        "removed1pct",
                        "removed10pct",
                        # "removed50pct",
                          ]]
    labels = ["true", "pred"]
    betas = []
    results = {}
    ratios = {}

    if len(samples) == 0 or len(metrics) == 0 or len(labels) == 0:
        print("What did you even expect to happen? Invalid input set.")
        return
    
    results[samples[0]] = dp.aggregate_data(samples[0], particleEnergy, species, layer_positions, file_limit, pred_cluster_cutoff=False)
    if len(samples) > 1:
        for sample in samples[1]:
            results[sample] = dp.aggregate_data(sample, particleEnergy, species, layer_positions, file_limit, pred_cluster_cutoff=False)

    if betas != []:
        for beta in betas:
            beta_sample_name = f"{samples[0]}_{int(beta*100)}"
            print(beta,beta_sample_name)
            samples[1].append(beta_sample_name)
            results[beta_sample_name] = dp.aggregate_data(samples[0], particleEnergy, species, layer_positions, file_limit, pred_cluster_cutoff=False, threshold_beta=beta)

    # produce the set of graph combinations
    # 1. sampleVar vs base true/true
    # 2. sampleVar vs base pred/pred
    # Note, Choosing sig=0 will not allow for the ratio of ratios plots. 

    sampleCombos = list(combinations_with_replacement(samples, 2))
    labelCombos = list(combinations_with_replacement(labels, 2))
    fullCombos = list(product(sampleCombos,labelCombos))
    for metric in metrics:
        ratios[metric] = {}
        for combo in fullCombos:
            sampleNum,sampleDen = combo[0][0],combo[0][1]
            labelNum,labelDen   = combo[1][0],combo[1][1]

            #kill undesirable combos
            if (sampleNum == sampleDen and labelNum == labelDen): continue
            if (sampleNum != sampleDen and labelNum != labelDen): continue
            #process remainer for case of list or not
            if (type(sampleDen) == str):
                # print(f"Plotting1 {metric}: {sampleNum} | {sampleDen} | {labelNum} | {labelDen}")
                ratios[metric][sampleDen] = plot_ratio(results, metric, (sampleNum,labelNum), (sampleDen,labelDen), species, pred_cluster_cutoff=False)
            else:
                if (type(sampleNum) == str):
                    for compSample in sampleDen:
                            # print(f"Plotting2 {metric}: {sampleNum} | {compSample} | {labelNum} | {labelDen}")
                            plot_ratio(results, metric, (sampleNum,labelNum), (compSample,labelDen),species, pred_cluster_cutoff=False)
                else:
                    for compSample in sampleNum:
                        # print(f"Plotting3 {metric}: {compSample} | {compSample} | {labelNum} | {labelDen}")
                        ratiosTemp = plot_ratio(results, metric, (compSample,labelNum), (compSample,labelDen), species, pred_cluster_cutoff=False)
                        rofR(ratiosTemp, *ratios[metric].values(), metric, compSample, *ratios[metric].keys(), species, pred_cluster_cutoff=False) #ratio2 should be the base sample


    # print(f"Plotting energy_res {samples[0]}")
    # plot_energy_resolution(results[samples[0]], samples[0])
    # if (len(samples) > 1):
    #     for sample in samples[1]:
    #         print(f"Plotting energy_res {sample}")
    #         plot_energy_resolution(results[sample], sample)
            
    
if __name__ == '__main__':
    main()
