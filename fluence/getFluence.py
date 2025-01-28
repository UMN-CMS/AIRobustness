#!/usr/bin/env python

import pickle as pkl

inputFile = 'HLLHCFluenceData/fluenceInterpolator.pkl'

with open(inputFile,'rb') as f: fluenceInterpolator = pkl.load(f)

# Example usage
z = 251.52
R = 4.08
print('z = {} cm, R = {} cm, fluence = {} HEH/cm^2'.format(z,R,fluenceInterpolator((R,z))))
