#!/usr/bin/env python3

import argparse
import os
import glob
import math

slurmDir = '../slurmFiles'

os.system('rm -rf config {}'.format(slurmDir))
os.system('mkdir config {}'.format(slurmDir))

parser = argparse.ArgumentParser(description="Slurm submit script")
parser.add_argument('-n','--nEvents',type=int,required=True,help='Number of events per job')
parser.add_argument('-i','--input',type=str,required=True,help='Path to directory with npz files')
parser.add_argument('-t','--tag',type=str,default='test',help='Tag for output')
args=parser.parse_args()

nEvents = args.nEvents
inputDir = args.input
tag = args.tag

inputFiles = sorted(glob.glob('{}/*'.format(inputDir)))
nJobs = math.ceil(len(inputFiles) / nEvents)
print('Submitting {} jobs...'.format(nJobs))
for iJob in range(nJobs):
    jobFiles = inputFiles[nEvents*iJob:nEvents*(iJob+1)]
    jobList = ' '.join(jobFiles).replace('/','\/')
    os.system('cp templates/submit.slurm config/submit{}.slurm'.format(iJob))
    os.system('sed -i -e \"s/INPUT/{}/g\" -e \"s/TAG/{}/g\" -e \"s/NJOB/{}/g\" config/submit{}.slurm'.format(jobList,tag,iJob,iJob))
    os.system('sbatch -p msismall config/submit{}.slurm'.format(iJob))
