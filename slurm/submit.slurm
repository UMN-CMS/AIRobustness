#!/bin/bash -l        
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2g
#SBATCH --tmp=2g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XXX@umn.edu # Email if you want notifications

cd $OLDHOME/XXX # Directory with singularity container (.sif), conda environment (env), and hgcal_minimal_eval_example (on branch pickleDev)
echo "#!/bin/bash -l
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
source env/bin/activate
cd hgcal_minimal_eval_example
python plot3d.py --pickle --input XXX --tag XXX
" > singRun.sh

# --input is the directory with the npz files, --tag is what it will name the output

chmod 755 singRun.sh

module load singularity
singFile=$PWD/pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
export SINGULARITY_BIND="$PWD:/wd"
singularity exec --bind $PWD:$PWD $singFile $PWD/singRun.sh
