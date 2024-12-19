import numpy as np
import plot_labels
import hist
import matplotlib.pyplot as plt


def plot_ratio(data, metric, numerator, denominator, sig=None):
    '''
    Generic plotting utility for making histogram ratio comparison plots.
    hist1 and hist2 must be 1d arrays of data to be turned into histograms
    formatText is a dict containing all graphs strings for formatting. 
    If anything isnt present, it will default to a generic string.
    '''
    # 
    # If defined, the plot is centered on the average and given sig stds away on either side
    # Reccommended to use sig=3 or more, typically sig=5 if theres little outliers
    # compare can be set to either "pred" or "true" if we want to compare either the truths or the preds for the two data sets
    # Extract necessary data

    data1 = data[numerator[0]][f"{metric}_{numerator[1]}"]
    data2 = data[denominator[0]][f"{metric}_{denominator[1]}"]

    formatText = plot_labels.plot_labels_select(metric, numerator, denominator)


    # binning
    avg = np.mean([np.mean(data1), np.mean(data2)]) # Plots center of both hists
    std = np.mean([np.std(data1),np.std(data2)])

    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[3,1])
    axs = gs.subplots(sharex=True)
    fig.suptitle(formatText["title"])

    hist1 = np.histogram(data1,bins=formatText["bins"],range=(formatText["lower"],formatText["upper"]))
    hist2 = np.histogram(data2,bins=formatText["bins"],range=(formatText["lower"],formatText["upper"]))

    axs[0].hist(data1,bins=formatText["bins"],range=(formatText["lower"],formatText["upper"]), histtype="step")
    axs[0].hist(data2,bins=formatText["bins"],range=(formatText["lower"],formatText["upper"]), histtype="step")

    ratiox = [hist1[1][x] for x in range(len(hist1[0])) if hist1[0][x]!=0 and hist2[0][x]!=0]
    ratioy = [hist1[0][x]/hist2[0][x] for x in range(len(hist1[0])) if hist1[0][x]!=0 and hist2[0][x]!=0]
    axs[1].scatter(ratiox,ratioy)
    axs[1].plot([formatText["lower"],formatText["upper"]], [1,1], linestyle='dotted', marker='o')
    axs[1].set_ylim([0,2])
    
    # if sig==0:
    #     upper = np.max(np.concatenate((data1,data2)))
    #     lower = np.min(np.concatenate((data1,data2)))
    #     hist_1 = hist.Hist(
    #         hist.axis.Regular(
    #             100, lower-0.5*std, upper+0.5*std, # add a buffer to the bounds
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data1)

    #     hist_2 = hist.Hist(
    #         hist.axis.Regular(
    #             100, lower-0.5*std, upper+0.5*std,
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data2)
    # elif sig==None:
    #     hist_1 = hist.Hist(
    #         hist.axis.Regular(
    #             formatText["bins"], formatText["lower"], formatText["upper"],
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data1)

    #     hist_2 = hist.Hist(
    #         hist.axis.Regular(
    #             formatText["bins"], formatText["lower"], formatText["upper"],
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data2)
    # else:
    #     hist_1 = hist.Hist(
    #         hist.axis.Regular(
    #             100, avg-sig*std, avg+sig*std,
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data1)

    #     hist_2 = hist.Hist(
    #         hist.axis.Regular(
    #             100, avg-sig*std, avg+sig*std,
    #             label=formatText["x_axis"], underflow=False, overflow=False
    #         )
    #     ).fill(data2)

    # fig = plt.figure(figsize=(10,8))
    # plt.axis("off") #When adding a title, it draws an entire figure, We only want the title
    
    # main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
    #     hist_2,
    #     rp_ylabel=formatText["y_axis"],
    #     # rp_ylim=[0,2],
    #     rp_num_label=f"{formatText['label1']}, $\mu$ {np.mean(data1):.2f}, $\sigma$ {np.std(data1):.2f}",
    #     rp_denom_label=f"{formatText['label2']}, $\mu$ {np.mean(data2):.2f}, $\sigma$ {np.std(data2):.2f}",
    #     rp_uncert_draw_type="bar",  # line or bar
        
    # )
    
    fig.savefig(formatText["saveas"])
    plt.close()
    return


def main():

    results = {}
    results["nominal"] = {"bestFit_r95_true": np.random.normal(6,0.5,1000),
                          "bestFit_r95_pred": np.random.normal(6,0.5,1000)}
    plot_ratio(results, "bestFit_r95", ("nominal","true"), ("nominal","pred"))

if __name__=="__main__":
    main()