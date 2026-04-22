# Minimal example for HGCAL evaluation

## Setup (for CPU)

```bash
# Download the model weights
xrdcp root://cmseos.fnal.gov//store/user/klijnsma/hgcal/ckpts/ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar .

# Download the data and extract
xrdcp root://cmseos.fnal.gov//store/user/klijnsma/hgcal/taus_2021_v1.tar.gz .
tar xf taus_2021_v1.tar.gz

# Download the singularity container
xrdcp root://cmseos.fnal.gov//store/user/klijnsma/hgcal/pytorch_2.0.0-cuda11.7-cudnn8-devel.sif .

# Clone necessary repositories
git clone -b oc_cuda git@github.com:tklijnsma/pytorch_cmspepr.git
git clone git@github.com:tklijnsma/cmspepr_hgcal_core.git

# Boot up the container, binding current wd to /wd
singularity run --bind $PWD:/wd pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
```

Once inside the container:

```bash
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
cd /wd
python -m venv env
source env/bin/activate

# Install torch_geometric with extensions
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other standard packages
pip install matplotlib plotly tqdm

# Install the kNN/OC extensions for CPU
pip install -e pytorch_cmspepr/
pip install -e cmspepr_hgcal_core/
```

Note that pytorch is preinstalled in the container.

```bash
python plot3d.py
```

This script will create a file called `myplots.html`, which can be opened in a browser.


# Plotting
## Environment
All plotting utilities use the 'airobust' conda environment given in airobust_environment.yml.
Create and activate the the environment with
```bash
conda create -n airobust --file airobust_environment.yml
conda activate airobust
```

**Paper Plot:** Scripts used to create specific plots for the AIRobustness paper.
- energy_resolution.py
- rofR_paper.py

The remainder scripts are for handling the network output and generic plotting utilities.

Several scripts reference event files in ```AggregatedPlottingScripts/pickles/<species>/<energy>/<sample>/``` which is an event list put in column major order by the script process_pkl.py for a moderate decrease in loading time. To read from a pkl event list direcly, replace instances of load_bulk_data() with load_data().

