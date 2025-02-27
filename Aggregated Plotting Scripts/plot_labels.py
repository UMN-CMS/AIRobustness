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
    formatText["title"]  = get_title(metric, sampleNum, sampleDen)
    formatText["x_axis"] = get_x_axis(metric)
    formatText["y_axis"] = "Ratio"
    formatText["label1"] = get_label(metric, sampleNum, sampleDen, labelNum)
    formatText["label2"] = get_label(metric, sampleDen, sampleNum, labelDen)
    formatText["saveas"] = get_saveName(metric, sampleNum, sampleDen, labelNum, labelDen)
    formatText["lower"], formatText["upper"], formatText["bins"] = get_range(metric, sampleNum)
    formatText["sig"]    = None

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

def get_saveName(metric, sample1, sample2, label1, label2):
    saveas = metric
    
    if sample1 != sample2:
        if label1 == label2:
            saveas += f"_{label1}s_{sample1}{sample2}"
        else:
            saveas += f"_{label1}{label2}_{sample1}{sample2}"
    else:
        saveas += f"_{sample1}"
    
    saveas += ".png"
    return saveas

def get_range(metric, sample):
    
    if sample in ["nominal","FTFP","singlePhotonLayer9","singlePhotonLayer8-9-10","singlePhotonZShift"]:
        return {
            "bestFit_r95":       (3, 9, 70),
            "bestFit_r68":       (0.7, 2.2, 70),
            "longitudinal_95":   (9, 18, 90),
            "longitudinal_68":   (4, 9.5, 88),
            "chi2":              (10, 55, 90),
            "radial_68":         (1.25, 3.75, 90),
            "radial_95":         (4.5, 9.5, 90),
            "abs_dists":         (1050, 2250, 80),
            "avg_weighted_dist": (1.25, 2.75, 80),
            "firstLayer":        (0, 50, 50),
            "maxELayer":         (0, 50, 50),
            "coe_layers":        (0, 50, 50),
        }[metric]
    elif sample in ["PionE50", "PionE50Layer29", "PionE50Neighbors"]:
        return {
            "bestFit_r95":       (0, 100, 100),
            "bestFit_r68":       (0, 40, 80),
            "longitudinal_95":   (0, 140, 70),
            "longitudinal_68":   (0, 70, 70),
            "chi2":              (0, 350, 70),
            "radial_68":         (0, 30, 60),
            "radial_95":         (0, 90, 90),
            "abs_dists":         (0, 7000, 80),
            "avg_weighted_dist": (0, 25, 75),
            "firstLayer":        (0, 50, 50),
            "maxELayer":         (0, 50, 50),
            "coe_layers":        (0, 50, 50),
        }[metric]
    elif sample in ["KaonE50", "KaonE50Layer29", "KaonE50Neighbors"]:
        return {
            "bestFit_r95":       (0, 100, 100),
            "bestFit_r68":       (0, 30, 90),
            "longitudinal_95":   (0, 140, 70),
            "longitudinal_68":   (0, 70, 70),
            "chi2":              (0, 350, 70),
            "radial_68":         (0, 30, 60),
            "radial_95":         (0, 100, 100),
            "abs_dists":         (0, 7000, 80),
            "avg_weighted_dist": (0, 25, 75),
            "firstLayer":        (0, 50, 50),
            "maxELayer":         (0, 50, 50),
            "coe_layers":        (0, 50, 50),
        }[metric]
    elif sample in ["TauE50", "TauE50Layer8", "TauE50Neighbors"]:
        return {
            "bestFit_r95":       (0, 80, 80),
            "bestFit_r68":       (0, 25, 75),
            "longitudinal_95":   (0, 130, 65),
            "longitudinal_68":   (0, 70, 70),
            "chi2":              (0, 300, 60),
            "radial_68":         (0, 30, 60),
            "radial_95":         (0, 80, 80),
            "abs_dists":         (0, 6000, 80),
            "avg_weighted_dist": (0, 25, 75),
            "firstLayer":        (0, 50, 50),
            "maxELayer":         (0, 50, 50),
            "coe_layers":        (0, 50, 50),
        }[metric]

"""
                                                                                    ARM
Python --interpreted-by-> C -compiled-to-> ASSEMBLY --runs-on-> (x86 frontend -> uOp cache -> uOp CPU) -GIZMOS> comp


"""
