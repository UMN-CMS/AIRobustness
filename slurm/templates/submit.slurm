#!/bin/bash -l        
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2g
#SBATCH --tmp=2g
#SBATCH --mail-type=NONE

cd $OLDHOME/hgcalmlSingularity

echo "#!/bin/bash -l
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
source env/bin/activate
cd hgcal_minimal_eval_example
python plot3d.py --pickle --input INPUT --tag TAG
" > slurmFiles/singRun_TAG_NJOB.sh

chmod 755 slurmFiles/singRun_TAG_NJOB.sh

module load singularity
singFile=$PWD/pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
export SINGULARITY_BIND="$PWD:/wd"
singularity exec --bind $PWD:$PWD $singFile $PWD/slurmFiles/singRun_TAG_NJOB.sh
