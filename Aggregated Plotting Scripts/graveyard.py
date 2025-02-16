#Nothing in here should run, this is just to contain old code before we decide we dont need it anymore
#I wouldnt anticipate this ever making a comeback since it is pretty much entirely replaced with the 
#other plotting scheme that is in place now.
#So dont lend this much more thought. If we need its in the old github submissions.

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
