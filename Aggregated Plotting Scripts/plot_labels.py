def default_format():
    formatText = {
        "title":"Hist Ratio",
        "x_axis":"X",
        "y_axis":"Y",
        "label":"Hist",
        "label1":"Hist1",
        "label2":"Hist2",
        "saveas":"histPlot.png",
        "sig":None
    }
    return formatText

def plot_labels_select(metric, numerator, denominator):
    sampleNum, labelNum = numerator[0], numerator[1]
    sampleDen, labelDen = denominator[0], denominator[1]
    formatText = {}
    
    title = get_title(metric, sampleNum, sampleDen)
    x_axis = get_x_axis(metric)
    y_axis = "Ratio"
    label1 = get_label(metric, sampleNum, sampleDen, labelNum)
    label2 = get_label(metric, sampleDen, sampleNum, labelDen)
    saveas = get_saveName(metric, sampleNum, sampleDen, labelNum)
    sig = None

    formatText["title"] = title
    formatText["x_axis"] = x_axis
    formatText["y_axis"] = y_axis
    formatText["label1"] = label1
    formatText["label2"] = label2
    formatText["saveas"] = saveas
    formatText["sig"] = sig
    
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
    else:
        print("Defaulting metric_title")
        title += default_format()["title"]

    return title

def get_x_axis(metric):
    if metric == "bestFit_r95" or metric == "bestFit_r68" or metric == "radial_68" or metric == "radial_95":
        return "Radial Distance (cm)"
    elif metric == "coe_layers":
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
    else:
        formatLabel += default_format()["label"]

    return formatLabel

def get_saveName(metric, sample1, sample2, label1):
    saveas = metric
    
    if sample1 != sample2:
        saveas += f"_{label1}_{sample1}{sample2}"
    else:
        saveas += f"_{sample1}"
    
    saveas += ".pdf"
    return saveas

