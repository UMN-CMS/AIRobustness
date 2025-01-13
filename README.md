# Minimal example for HGCAL evaluation

## Setup (every time)

```bash
singularity run --bind $PWD:/wd pytorch_2.0.0-cuda11.7-cudnn8-devel.sif

```

Once inside the container:

```bash
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
cd /wd
source env/bin/activate
```

## Initial setup (for CPU)

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

## Usage

```bash
python plot3d.py
```

This script will create a file called `myplots.html`, which can be opened in a browser.

## Running slurm jobs for model outputs

First, create a new directory in which to run your jobs within the directory with your container (.sif) file, and then copy over the contents of the `slurm` directory.
For example:

```
cd hgcalmlSingularity # Contains .sif file
mkdir slurmSinglePhotonExample25-01-13
cd slurmSinglePhotonExample25-01-13
cp -r ../hgcal_minimal_eval_example/slurm/* . 
```

`submitSlurm.py` handles division of the jobs and submission.
It has several command-line arguments:
- `--input` (or `-i`), which points to the directory with the npz files (this is a required argument)
- `--nEvents` (or `-n`), which is the number of events you would like per job (this is a required argument)
- `--tag` (or `-t`), which is the name of the directory in `output` to which the pkl files will be output

Note: running `./submitSlurm.py --help` will print out a help menu that explains all options. 

As an example, 
the following will take all .npz files in the directory `../hgcal_minimal_eval_example/singlePhotonExample/`, 
split them into jobs of 20 events each, 
and then submit them via slurm, placing the output in `hgcal_minimal_eval_example/output/test25-01-13`:

```
./submitSlurm.py -nEvents 20 -input ../hgcal_minimal_eval_example/singlePhotonExample/ -tag test25-01-13
```

To check the status of your jobs, run the following inserting your MSI username: 

```
squeue -u USER
```
