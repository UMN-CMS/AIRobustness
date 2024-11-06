import matplotlib.pyplot as plt
import glob
import plotly.graph_objects as go
import hist
from tqdm import tqdm
import numpy as np

import data_processing as dp

def aggregate_data(file_pattern, layer_positions, file_limit=1000):
    """Aggregate all necessary data from the files."""
    files = glob.glob(file_pattern)[:file_limit]
    results = {
        "radial_68_pred": [],
        "radial_95_pred": [],
        "radial_68_true": [],
        "radial_95_true": [],
        "coe_layers_pred": [],  # Changed from com_layers_pred
        "coe_layers_true": [],  # Changed from com_layers_true
        "longitudinal_68_pred": [],
        "longitudinal_95_pred": [],
        "longitudinal_68_true": [],
        "longitudinal_95_true": [],
        "chi2_true": [],
        "chi2_pred": [],
        "abs_dists_true": [],
        "abs_dists_pred": [],
        "bestFit_r95_true": [], # similar to radial_95 but using 3dbestFit line
        "bestFit_r95_pred": [],
        "bestFit_r68_true": [],
        "bestFit_r68_pred": [],
        "avg_weighted_dist_true": [],
        "avg_weighted_dist_pred": []
    }

    for file_path in tqdm(files):
        data, score_noise_filter, pass_noise_filter, out_gravnet = dp.load_data(file_path)
        true_energies, true_clusters, xpos, ypos, zpos = dp.process_data(data)
        final_pred_hits = dp.process_gravnet(score_noise_filter, pass_noise_filter, out_gravnet)

        # Radial Shower Spread calculations
        valid_pred_indices = np.where((final_pred_hits != -1) & (final_pred_hits != 0) & (final_pred_hits != -2))[0]
        valid_true_indices = np.where(true_clusters != 0)[0]

        if len(valid_pred_indices) > 0:
            radial_68_pred, radial_95_pred = dp.calculate_radial_shower_spread(valid_pred_indices, xpos, ypos, true_energies)
            results["radial_68_pred"].append(radial_68_pred)
            results["radial_95_pred"].append(radial_95_pred)

        if len(valid_true_indices) > 0:
            radial_68_true, radial_95_true = dp.calculate_radial_shower_spread(valid_true_indices, xpos, ypos, true_energies)
            results["radial_68_true"].append(radial_68_true)
            results["radial_95_true"].append(radial_95_true)

        # Longitudinal Shower Spread and COE layers calculations
        if len(valid_pred_indices) > 0:
            coe_layer_pred, longitudinal_68_pred, longitudinal_95_pred = dp.calculate_longitudinal_shower_spread(
                valid_pred_indices, zpos, true_energies, layer_positions)
            results["coe_layers_pred"].append(coe_layer_pred)
            results["longitudinal_68_pred"].append(longitudinal_68_pred)
            results["longitudinal_95_pred"].append(longitudinal_95_pred)

        if len(valid_true_indices) > 0:
            coe_layer_true, longitudinal_68_true, longitudinal_95_true = dp.calculate_longitudinal_shower_spread(
                valid_true_indices, zpos, true_energies, layer_positions)
            results["coe_layers_true"].append(coe_layer_true)
            results["longitudinal_68_true"].append(longitudinal_68_true)
            results["longitudinal_95_true"].append(longitudinal_95_true)

        #process inputs for noise
        x_true = xpos[true_clusters==1]
        y_true = ypos[true_clusters==1]
        z_true = zpos[true_clusters==1]
        energy_true = true_energies[true_clusters==1]
        x_pred = xpos[final_pred_hits > 0]
        y_pred = ypos[final_pred_hits > 0]
        z_pred = zpos[final_pred_hits > 0]
        energy_pred = true_energies[final_pred_hits > 0]

        abs_dists_true = dp.calculate_absolute_distances(x_true, y_true, z_true)
        abs_dists_pred = dp.calculate_absolute_distances(x_pred, y_pred, z_pred)
        
        results["chi2_true"].append(dp.calculate_chi2(abs_dists_true, energy_true))
        results["chi2_pred"].append(dp.calculate_chi2(abs_dists_pred, energy_pred))
        results["bestFit_r95_true"].append(dp.e_radius(abs_dists_true, energy_true, 0.95))
        results["bestFit_r68_true"].append(dp.e_radius(abs_dists_true, energy_true, 0.68))
        results["bestFit_r95_pred"].append(dp.e_radius(abs_dists_pred, energy_pred, 0.95))
        results["bestFit_r68_pred"].append(dp.e_radius(abs_dists_pred, energy_pred, 0.68))
        results["avg_weighted_dist_true"].append(sum(abs_dists_true * energy_true) / sum(energy_true))
        results["avg_weighted_dist_pred"].append(sum(abs_dists_pred * energy_pred) / sum(energy_pred))

    return results

def plot_radial_shower_spread(results):
    """Plot histograms for Radial Shower Spread."""
    plt.figure(figsize=(18, 6))

    # Determine the range for the bins based on both 68% and 95% data
    all_radial_values_pred = np.concatenate([results["radial_68_pred"], results["radial_95_pred"]])
    all_radial_values_true = np.concatenate([results["radial_68_true"], results["radial_95_true"]])
    min_val = min(np.min(all_radial_values_pred), np.min(all_radial_values_true))
    max_val = max(np.max(all_radial_values_pred), np.max(all_radial_values_true))

    # Create bins with smaller bin size (0.1)
    bin_size = 0.1
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Plot predicted radial spread
    plt.subplot(1, 2, 1)
    plt.hist(results["radial_68_pred"], bins=bins, alpha=0.5, label='68% Pred Radial Spread')
    plt.hist(results["radial_95_pred"], bins=bins, alpha=0.5, label='95% Pred Radial Spread')
    plt.xlabel('Radial Distance')
    plt.ylabel('Frequency')
    plt.title('Predicted Radial Shower Spread')
    plt.legend()

    # Plot true radial spread
    plt.subplot(1, 2, 2)
    plt.hist(results["radial_68_true"], bins=bins, alpha=0.5, label='68% True Radial Spread')
    plt.hist(results["radial_95_true"], bins=bins, alpha=0.5, label='95% True Radial Spread')
    plt.xlabel('Radial Distance')
    plt.ylabel('Frequency')
    plt.title('True Radial Shower Spread')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_longitudinal_shower_spread(results, layer_positions):
    """Plot histograms for Longitudinal Shower Spread and COE layers."""
    plt.figure(figsize=(18, 6))

    # Determine the range for the bins based on both 68% and 95% data
    all_longitudinal_values_pred = np.concatenate([results["longitudinal_68_pred"], results["longitudinal_95_pred"]])
    all_longitudinal_values_true = np.concatenate([results["longitudinal_68_true"], results["longitudinal_95_true"]])
    min_val = min(np.min(all_longitudinal_values_pred), np.min(all_longitudinal_values_true))
    max_val = max(np.max(all_longitudinal_values_pred), np.max(all_longitudinal_values_true))

    # Create bins with smaller bin size (0.1)
    bin_size = 0.1
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Plot predicted longitudinal spread
    plt.subplot(1, 2, 1)
    plt.hist(results["longitudinal_68_pred"], bins=bins, alpha=0.5, label='68% Pred Longitudinal Spread')
    plt.hist(results["longitudinal_95_pred"], bins=bins, alpha=0.5, label='95% Pred Longitudinal Spread')
    plt.xlabel('Longitudinal Spread (cm)')
    plt.ylabel('Number of Events')
    plt.title('Predicted Longitudinal Shower Spread')
    plt.legend()
    plt.yscale('log')

    # Plot true longitudinal spread
    plt.subplot(1, 2, 2)
    plt.hist(results["longitudinal_68_true"], bins=bins, alpha=0.5, label='68% True Longitudinal Spread')
    plt.hist(results["longitudinal_95_true"], bins=bins, alpha=0.5, label='95% True Longitudinal Spread')
    plt.xlabel('Longitudinal Spread (cm)')
    plt.ylabel('Number of Events')
    plt.title('True Longitudinal Shower Spread')
    plt.legend()
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

def plot_coe_layers(results, layer_positions):
    """Plot histograms for COE layers."""
    plt.figure(figsize=(18, 6))

    # Plot COE layers for predicted
    plt.subplot(1, 2, 1)
    counts, _ = np.histogram(results["coe_layers_pred"], bins=np.arange(1, len(layer_positions) + 2) - 0.5)
    plt.hist(results["coe_layers_pred"], bins=np.arange(1, len(layer_positions) + 2) - 0.5, alpha=0.5, label='Predicted COE Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Events')
    plt.title('Predicted COE Layers')
    plt.legend()    
    plt.yscale('log')


    # Plot COE layers for true
    plt.subplot(1, 2, 2)
    counts, _ = np.histogram(results["coe_layers_true"], bins=np.arange(1, len(layer_positions) + 2) - 0.5)
    plt.hist(results["coe_layers_true"], bins=np.arange(1, len(layer_positions) + 2) - 0.5, alpha=0.5, label='True COE Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Number of Events')
    plt.title('True COE Layers')
    plt.legend()
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

def plot_radius_95(results, sig=None):
    # If sig is none, the graph is not zoomed
    # If defined, the plot is centered on the average and given sig stds away on either side
    # Reccommended to use sig=3 or more, typically sig=5 if theres little outliers
    # Extract necessary data
    r_true = results["bestFit_r95_true"]
    r_pred = results["bestFit_r95_pred"]
    # binning
    avg = np.mean([np.mean(r_true), np.mean(r_pred)]) # Plots center of both hists
    std = np.mean([np.std(r_true),np.std(r_pred)])

    if sig==None:
        upper = np.max(np.concatenate((r_true,r_pred)))
        lower = np.min(np.concatenate((r_true,r_pred)))
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std, # add a buffer to the bounds
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_pred)
    else:
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_pred)

    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
        hist_2,
        rp_ylabel=r"Ratio",
        rp_num_label=f"True 95%, $\mu$ {np.mean(r_true):.2f}, $\sigma$ {np.std(r_true):.2f}",
        rp_denom_label=f"Pred 95%, $\mu$ {np.mean(r_pred):.2f}, $\sigma$ {np.std(r_pred):.2f}",
        rp_uncert_draw_type="bar",  # line or bar
    )
    fig.savefig("bestFitRadius95.png")

def plot_radius_68(results, sig=None):
    # If sig is none, the graph is not zoomed
    # If defined, the plot is centered on the average and given sig stds away on either side
    # Reccommended to use sig=3 or more, typically sig=5 if theres little outliers
    # Extract necessary data
    r_true = results["bestFit_r68_true"]
    r_pred = results["bestFit_r68_pred"]

    avg = np.mean([np.mean(r_true), np.mean(r_pred)]) # Plots center of both hists
    std = np.mean([np.std(r_true),np.std(r_pred)])

    if sig==None:
        upper = np.max(np.concatenate((r_true,r_pred)))
        lower = np.min(np.concatenate((r_true,r_pred)))
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std, # add a buffer to the bounds
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_pred)
    else:
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="Radius (cm)", underflow=False, overflow=False
            )
        ).fill(r_pred)

    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
        hist_2,
        rp_ylabel=r"Ratio",
        rp_num_label=f"True 68%, $\mu$ {np.mean(r_true):.2f}, $\sigma$ {np.std(r_true):.2f}",
        rp_denom_label=f"Pred 68%, $\mu$ {np.mean(r_pred):.2f}, $\sigma$ {np.std(r_pred):.2f}",
        rp_uncert_draw_type="bar",  # line or bar
    )
    fig.savefig("bestFitRadius68.png")

def plot_chi2(results, sig=None):
    # If sig is none, the graph is not zoomed
    # If defined, the plot is centered on the average and given sig stds away on either side
    # Reccommended to use sig=3 or more, typically sig=5 if theres little outliers
    # Extract necessary data
    chi_true = results["chi2_true"]
    chi_pred = results["chi2_pred"]

    avg = np.mean([np.mean(chi_true),np.mean(chi_pred)]) # Plots center of both hists
    std = np.mean([np.std(chi_true), np.std(chi_pred)])

    if sig==None:
        upper = np.max(np.concatenate((chi_true,chi_pred)))
        lower = np.min(np.concatenate((chi_true,chi_pred)))
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std, # add a buffer to the bounds
                name="X", label="chi2", underflow=False, overflow=False
            )
        ).fill(chi_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std,
                name="X", label="chi2", underflow=False, overflow=False
            )
        ).fill(chi_pred)
    else:
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="chi2", underflow=False, overflow=False
            )
        ).fill(chi_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="chi2", underflow=False, overflow=False
            )
        ).fill(chi_pred)

    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
        hist_2,
        rp_ylabel=r"Ratio",
        rp_num_label=f"True chi2, $\mu$ {np.mean(chi_true):.2f}, $\sigma$ {np.std(chi_true):.2f}",
        rp_denom_label=f"Pred chi2, $\mu$ {np.mean(chi_pred):.2f}, $\sigma$ {np.std(chi_pred):.2f}",
        rp_uncert_draw_type="bar",  # line or bar
    )
    fig.savefig("chi2.png")

def plot_avg_weighted_dists(results, sig=None):
    # If sig is none, the graph is not zoomed
    # If defined, the plot is centered on the average and given sig stds away on either side
    # Reccommended to use sig=3 or more, typically sig=5 if theres little outliers
    # Extract necessary data
    dists_true = results["avg_weighted_dist_true"]
    dists_pred = results["avg_weighted_dist_pred"]

    avg = np.mean([np.mean(dists_true),np.mean(dists_pred)]) # Plots center of both hists
    std = np.mean([np.std(dists_true), np.std(dists_pred)])

    if sig==None:
        upper = np.max(np.concatenate((dists_true,dists_pred)))
        lower = np.min(np.concatenate((dists_true,dists_pred)))
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std, # add a buffer to the bounds
                name="X", label="avg_dist", underflow=False, overflow=False
            )
        ).fill(dists_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, lower-0.5*std, upper+0.5*std,
                name="X", label="avg_dist", underflow=False, overflow=False
            )
        ).fill(dists_pred)
    else:
        hist_1 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="avg_dist", underflow=False, overflow=False
            )
        ).fill(dists_true)

        hist_2 = hist.Hist(
            hist.axis.Regular(
                100, avg-sig*std, avg+sig*std,
                name="X", label="avg_dist", underflow=False, overflow=False
            )
        ).fill(dists_pred)

    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    main_ax_artists, sublot_ax_arists = hist_1.plot_ratio(
        hist_2,
        rp_ylabel=r"Ratio",
        rp_num_label=f"True average weighted distance, $\mu$ {np.mean(dists_true):.2f}, $\sigma$ {np.std(dists_true):.2f}",
        rp_denom_label=f"Pred average weighted distance, $\mu$ {np.mean(dists_pred):.2f}, $\sigma$ {np.std(dists_pred):.2f}",
        rp_uncert_draw_type="bar",  # line or bar
    )
    fig.savefig("avg_dist.png")

def main():
    # Set file pattern and file limit
    # file_pattern = r'C:\Users\tsoli\OneDrive\Documents\School\1 - University of Minnesota\Year 17\Year 1 Research\picklefiles\photons\*.pkl'
    #MSI paths
    file_pattern = "/home/nstrobbe/mahon336/hgcalmlSingularity/hgcal_minimal_eval_example/output/singlePhoton24-04-01/nominal/*.pkl"
    # file_pattern = "/home/nstrobbe/mahon336/hgcalmlSingularity/hgcal_minimal_eval_example/output/singlePhoton24-04-01/FTFP_BERT_EMN/*.pkl"
    file_limit = 1000
    layer_positions = np.array([
        322, 323, 325, 326, 328, 329, 331, 332, 334, 335,
        337, 338, 340, 341, 343, 344, 346, 347, 349, 350,
        352, 353, 355, 356, 358, 359, 361, 362, 368, 373,
        379, 384, 389, 395, 400, 406, 411, 417, 422, 428,
        436, 445, 453, 462, 470, 479, 487, 496, 505, 513
    ])

    # Step 1: Aggregate data
    results = aggregate_data(file_pattern, layer_positions, file_limit)

    # Step 2: Plot different metrics
    # plot_radial_shower_spread(results)
    # plot_longitudinal_shower_spread(results, layer_positions)
    # plot_coe_layers(results, layer_positions)
    # plot_radius_95(results, 4)
    # plot_radius_68(results, 3)
    # plot_chi2(results, 4)
    plot_avg_weighted_dists(results, 4)


if __name__ == '__main__':
    main()
