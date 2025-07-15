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

def plot_labels_select(metric, numerator, denominator, species, pred_cluster_cutoff):
    sampleNum, labelNum = numerator[0], numerator[1]
    sampleDen, labelDen = denominator[0], denominator[1]
    
    formatText = {}
    formatText["title"]  = get_title(metric, sampleNum, sampleDen)
    formatText["x_axis"] = get_x_axis(metric)
    formatText["y_axis"] = "Ratio"
    formatText["label1"] = get_label(metric, sampleNum, sampleDen, labelNum)
    formatText["label2"] = get_label(metric, sampleDen, sampleNum, labelDen)
    formatText["saveas"] = get_saveName(metric, sampleNum, sampleDen, labelNum, labelDen)
    formatText["lower"], formatText["upper"], formatText["bins"] = get_range(metric, sampleNum, species, pred_cluster_cutoff)
    formatText["sig"]    = None

    return formatText

def get_title(metric, sampleNum, sampleDen):
    if sampleNum == sampleDen:
        title = f"Sample {sampleNum} "
    else:
        title = f"Comparison ({sampleNum},{sampleDen}) "

    if metric == "bestFit_r99":
        title += "BestFit Radius 99"
    elif metric == "bestFit_r95":
        title += "BestFit Radius 95"
    elif metric == "bestFit_r68":
        title += "BestFit Radius 68"
    elif metric == "bestFit_r99-95":
        title += "BestFit Radius 99-95"
    elif metric == "bestFit_r95-68":
        title += "BestFit Radius 95-68"
    elif metric == "coe_layers":
        title += "Center of Energy Layer"
    elif metric == "longitudinal_95":
        title += "Longitudinal spread 95"
    elif metric == "longitudinal_68":
        title += "Longitudinal Spread 68"
    elif metric == "chi2":
        title += "Chi2"
    elif metric == "emRatio":
        title += "EM Ratio"    
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
        print(f"Defaulting metric_title {metric}")
        title += default_format()["title"]

    return title

def get_x_axis(metric):
    if metric == "bestFit_r95" or metric == "bestFit_r68" or metric == "radial_68" or metric == "radial_95" or metric == "bestFit_r99" or metric == "bestFit_r99-95" or metric == "bestFit_r95-68":
        return "Radial Distance (cm)"
    elif metric == "coe_layers" or metric == "firstLayer" or metric == "maxELayer":
        return "Layer Index"
    elif metric == "longitudinal_68" or metric == "longitudinal_95":
        return "Longitudinal Distance (cm)"
    elif metric == "chi2":
        return "Chi2"
    elif metric == "emRatio":
        return "EM Energy / Total Energy"    
    elif metric == "abs_dists" or metric == "avg_weighted_dist":
        return "Distance (cm)"
    else:
        print(f"Defaulting x_axis {metric}")
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
    elif metric == "bestFit_r99":
        formatLabel += "99%"
    elif metric == "bestFit_r95" or metric == "radial_95" or metric == "longitudinal_95":
        formatLabel += "95%"
    elif metric == "bestFit_r68" or metric == "radial_68" or metric == "longitudinal_68":
        formatLabel += "68%"
    elif metric == "bestFit_r99-95":
        formatLabel += "99-95%"
    elif metric == "bestFit_r95-68":
        formatLabel += "95-68%"
    elif metric == "emRatio":
        formatLabel += "EM Ratio"
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
        print(f"Defaulting label {metric}")
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

def get_range(metric, sample, species, pred_cluster_cutoff):
    # Change to JSON some time. Might be worth
    if species == "Photon":
        return {
            "abs_dists":         ((1000,9000,50),(1050, 2250, 80)),
            "avg_weighted_dist": ((0,10,30),(1.25, 2.75, 80)),
            "bestFit_r68":       ((0,12,36),(0.7, 2.2, 70)),
            "bestFit_r95-68":    ((2,10,40),(0, 100, 50)),
            "bestFit_r95":       ((2.5,20,40),(3, 9, 70)),
            "bestFit_r99-95":    ((0,30,30),(0, 100, 50)),
            "bestFit_r99":       ((0,45,30),(0, 80, 40)),
            "chi2":              ((0,600,40),(10, 55, 90)),
            "coe_layers":        ((0,50,50),(0, 50, 50)),
            "emRatio":           ((0,1,30),(0,1,30)),
            "firstLayer":        ((0,50,50),(0, 50, 50)),
            "longitudinal_68":   ((4,11,40),(4, 9.5, 88)),
            "longitudinal_95":   ((8,20,40),(9, 18, 90)),
            "maxELayer":         ((0,50,50),(0, 50, 50)),
            "radial_68":         ((1,4.5,30),(1.25, 3.75, 90)),
            "radial_95":         ((0,14,40),(4.5, 9.5, 90)),
        }[metric][int(pred_cluster_cutoff)]
    elif species == "Pion":
        return {
            "abs_dists":         ((0,17500,50),(0, 7000, 40)),
            "avg_weighted_dist": ((0,40,40),(0, 25, 50)),
            "bestFit_r68":       ((0,50,40),(0, 40, 40)),
            "bestFit_r95-68":    ((0,110,40),(0, 100, 50)),
            "bestFit_r95":       ((0,130,40),(0, 80, 40)),
            "bestFit_r99-95":    ((0,100,40),(0, 100, 50)),
            "bestFit_r99":       ((0,175,40),(0, 100, 50)),
            "chi2":              ((0,1750,50),(0, 350, 35)),
            "coe_layers":        ((0,50,50),(0, 50, 50)),
            "emRatio":           ((0,1,30),(0,1,30)),
            "firstLayer":        ((0,50,50),(0, 50, 50)),
            "longitudinal_68":   ((0,70,35),(0, 70, 35)),
            "longitudinal_95":   ((0,140,35),(0, 140, 35)),
            "maxELayer":         ((0,50,50),(0, 50, 50)),
            "radial_68":         ((0,40,40),(0, 30, 30)),
            "radial_95":         ((0,100,40),(0, 90, 45)),
        }[metric][int(pred_cluster_cutoff)]
    elif species == "Kaon":
        return {
            "abs_dists":         ((0,17500,40),(0, 7000, 40)),
            "avg_weighted_dist": ((0,45,45),(0, 25, 50)),
            "bestFit_r68":       ((0,60,40),(0, 30, 45)),
            "bestFit_r95-68":    ((0,100,40),(0, 100, 50)),
            "bestFit_r95":       ((0,150,50),(0, 100, 50)),
            "bestFit_r99-95":    ((0,120,40),(0, 100, 50)),
            "bestFit_r99":       ((0,175,35),(0, 100, 50)),
            "chi2":              ((0,1500,50),(0, 350, 35)),
            "coe_layers":        ((0,50,50),(0, 50, 50)),
            "emRatio":           ((0,1,30),(0,1,30)),
            "firstLayer":        ((0,50,50),(0, 50, 50)),
            "longitudinal_68":   ((0,80,40),(0, 70, 35)),
            "longitudinal_95":   ((0,130,40),(0, 140, 35)),
            "maxELayer":         ((0,50,50),(0, 50, 50)),
            "radial_68":         ((0,30,30),(0, 30, 30)),
            "radial_95":         ((0,110,30),(0, 100, 50)),
        }[metric][int(pred_cluster_cutoff)]
    elif species == "Tau":
        return {
            "abs_dists":         ((0,6000,40),(0, 6000, 40)),
            "avg_weighted_dist": ((0,25,50),(0, 25, 50)),
            "bestFit_r68":       ((0,25,50),(0, 25, 50)),
            "bestFit_r95-68":    ((0,100,50),(0, 100, 50)),
            "bestFit_r95":       ((0,80,40),(0, 80, 40)),
            "bestFit_r99-95":    ((0,100,40),(0, 100, 50)),
            "bestFit_r99":       ((0,120,40),(0, 100, 50)),
            "chi2":              ((0,3000,30),(0, 300, 30)),
            "coe_layers":        ((0,50,50),(0, 50, 50)),
            "emRatio":           ((0,1,30),(0,1,30)),
            "firstLayer":        ((0,50,50),(0, 50, 50)),
            "longitudinal_68":   ((0,70,35),(0, 70, 35)),
            "longitudinal_95":   ((0,140,35),(0, 140, 35)),
            "maxELayer":         ((0,50,50),(0, 50, 50)),
            "radial_68":         ((0,30,30),(0, 30, 30)),
            "radial_95":         ((0,80,40),(0, 80, 40)),
            
        }[metric][int(pred_cluster_cutoff)]


