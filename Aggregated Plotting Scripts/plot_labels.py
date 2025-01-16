import numpy as np

def default_format():
    formatText = {
        "title":"Hist Ratio",
        "x_axis":"X",
        "y_axis":"Y",
        "label":"Hist",
        "label1":"Hist1",
        "label2":"Hist2",
        "saveas":"histPlot.png",
        "upper":100,
        "lower":0,
        "sig":None
    }
    return formatText

def plot_labels_select(metric, numerator, denominator):
    sampleNum, labelNum = numerator[0], numerator[1]
    sampleDen, labelDen = denominator[0], denominator[1]
    
    formatText = {}
    formatText["title"] = get_title(metric, sampleNum, sampleDen)
    formatText["x_axis"] = get_x_axis(metric)
    formatText["y_axis"] = "Ratio"
    formatText["label1"] = get_label(metric, sampleNum, sampleDen, labelNum)
    formatText["label2"] = get_label(metric, sampleDen, sampleNum, labelDen)
    formatText["saveas"] = get_saveName(metric, sampleNum, sampleDen, labelNum)
    formatText["bins"] = get_bins(metric)
    formatText["upper"] = get_upper(metric)
    formatText["lower"] = get_lower(metric)
    formatText["sig"] = None

    return formatText

def get_title(metric, sampleNum, sampleDen):
    if sampleNum == sampleDen:
        title = f"Sample {sampleNum} "
    else:
        title = f"Comparison ({sampleNum},{sampleDen}) "

    if metric == "bestFit_r95":
        title += "BestFit Radius 95"
    elif metric == "bestFit_r68":
        title += "BestFit Radius 68"
    elif metric == "coe_layers":
        title += "Center of Energy Layer"
    elif metric == "longitudinal_95":
        title += "Longitudinal spread 95"
    elif metric == "longitudinal_68":
        title += "Longitudinal Spread 68"
    elif metric == "chi2":
        title += "Chi2"
    elif metric == "radial_68":
        title += "Radial Spread 68"
    elif metric == "radial_95":
        title += "Radial Spread 95"
    elif metric == "abs_dists":
        title += "BestFit Absolute Distance"
    elif metric == "avg_weighted_dist":
        title += "BestFit Average Weighted Distance"
    elif metric == "firstLayer":
        title += "First Layer"
    elif metric == "maxELayer":
        title += "Max Energy Layer"
    else:
        print("Defaulting metric_title")
        title += default_format()["title"]

    return title

def get_x_axis(metric):
    if metric == "bestFit_r95" or metric == "bestFit_r68" or metric == "radial_68" or metric == "radial_95":
        return "Radial Distance (cm)"
    elif metric == "coe_layers" or metric == "firstLayer" or metric == "maxELayer":
        return "Layer Index"
    elif metric == "longitudinal_68" or metric == "longitudinal_95":
        return "Longitudinal Distance (cm)"
    elif metric == "chi2":
        return "Chi2"
    elif metric == "abs_dists" or metric == "avg_weighted_dist":
        return "Distance (cm)"
    else:
        print("Defaulting x_axis")
        return default_format()["x_axis"]
    
def get_label(metric, sample, sampleCompare, label):
    formatLabel = ""

    if label == "true":
        formatLabel += "True "
    elif label == "pred":
        formatLabel += "Pred "
    
    if sample != sampleCompare:
        formatLabel += f"{sample} "

    if metric == "chi2":
        formatLabel += "chi2"
    elif metric == "bestFit_r95" or metric == "radial_95" or metric == "longitudinal_95":
        formatLabel += "95%"
    elif metric == "bestFit_r68" or metric == "radial_68" or metric == "longitudinal_68":
        formatLabel += "68%"
    elif metric == "coe_layers":
        formatLabel += "COE layers"
    elif metric == "abs_dists":
        formatLabel += "Abs Distance"
    elif metric == "avg_weighted_dist":
        formatLabel += "Avg Weighted Distance"
    elif metric == "firstLayer":
        formatLabel += "layerIndex"
    elif metric == "maxELayer":
        formatLabel += "maxELayer"
    else:
        formatLabel += default_format()["label"]

    return formatLabel

def get_saveName(metric, sample1, sample2, label1):
    saveas = metric
    
    if sample1 != sample2:
        saveas += f"_{label1}_{sample1}{sample2}"
    else:
        saveas += f"_{sample1}"
    
    saveas += ".png"
    return saveas

def get_bins(metric):
    # in general I want some bin number between 50 and 100 but must be aligned to the integers 
    # so the difference must be a multiple of the divisor to get it close to 100 
    # this is kinda gross because it likely wont scale when we start getting different metrics but for now this will have to do
    # the sigma method seems a good alternative but maybe we can scale it to account for the asymmetric spread
    if metric == "bestFit_r95" : return 70
    elif metric == "bestFit_r68" : return 70
    elif metric == "coe_layers" : return 50
    elif metric == "longitudinal_95" : return 90
    elif metric == "longitudinal_68" : return 88
    elif metric == "chi2" : return 90
    elif metric == "radial_68" : return 90
    elif metric == "radial_95" : return 90
    elif metric == "abs_dists" : return 80
    elif metric == "avg_weighted_dist" : return 80
    elif metric == "firstLayer" : return 50
    elif metric == "maxELayer" : return 50
    
def get_upper(metric):
    if metric == "bestFit_r95" : return 9
    elif metric == "bestFit_r68" : return 2.2
    elif metric == "coe_layers" : return 50
    elif metric == "longitudinal_95" : return 18
    elif metric == "longitudinal_68" : return 9.5
    elif metric == "chi2" : return 55
    elif metric == "radial_68" : return 3.75
    elif metric == "radial_95" : return 9.5
    elif metric == "abs_dists" : return 2250
    elif metric == "avg_weighted_dist" : return 2.75
    elif metric == "firstLayer" : return 50
    elif metric == "maxELayer" : return 50
      
def get_lower(metric):
    if metric == "bestFit_r95" : return 3
    elif metric == "bestFit_r68" : return 0.7
    elif metric == "coe_layers" : return 0
    elif metric == "longitudinal_95" : return 9
    elif metric == "longitudinal_68" : return 4
    elif metric == "chi2" : return 10
    elif metric == "radial_68" : return 1.25
    elif metric == "radial_95" : return 4.5
    elif metric == "abs_dists" : return 1050
    elif metric == "avg_weighted_dist" : return 1.25
    elif metric == "firstLayer" : return 0
    elif metric == "maxELayer" : return 0