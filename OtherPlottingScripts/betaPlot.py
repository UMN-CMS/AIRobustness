import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from itertools import cycle
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import numpy as np
import argparse

# Select which plots you want it to do
doROCCurve = True
doSignalPlot = True

with open('data/betaTablesNew.pkl','rb') as f: data = pkl.load(f)

lines = ['-','--','-.',':']
lineCycler = cycle(lines)

colors = cm.rainbow(np.linspace(0,1,len(data['beta'].unique())))
colorCyclerBeta = cycle(colors)
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']  #'#a65628', '#f781bf', '#dede00', '#984ea3'
colorCyclerPart = cycle(colors)

data['FPR'] = data['ERecoSignal-ETrueNoise'] / (data['ERecoSignal-ETrueNoise'] + data['ERecoNoise-ETrueNoise'])
data['TPR'] = data['ERecoSignal-ETrueSignal'] / (data['ERecoSignal-ETrueSignal'] + data['ERecoNoise-ETrueSignal'])

labelDict = {   'Matched-EReco':r'Fraction of $E_{{\mathrm{{sig}}}}^{{\mathrm{{reco}}}}$ matched to true signal',
                'Matched-ETrue':r'Fraction of $E_{{\mathrm{{sig}}}}^{{\mathrm{{true}}}}$ matched to reco signal',
                'Unmatched-EReco':r'Fraction of $E_{{\mathrm{{sig}}}}^{{\mathrm{{reco}}}}$ matched to true noise',
                'Unmatched-ETrue':r'Fraction of $E_{{\mathrm{{sig}}}}^{{\mathrm{{true}}}}$ matched to reco noise',
             }

if doROCCurve:

    fig,ax = plt.subplots()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=2))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=0))

    for particle in data['particle'].unique():
        dataTemp = data[data['particle'] == particle]
        plt.plot(dataTemp['FPR'],dataTemp['TPR'],color='k',linestyle=next(lineCycler),label=particle,clip_on=False,zorder=100)

    for beta in data['beta'].unique():
        dataTemp = data[data['beta'] == beta]
        plt.plot(dataTemp['FPR'],dataTemp['TPR'],'.',markerfacecolor=next(colorCyclerBeta),markeredgecolor='k',markersize=10,label=rf'$\beta = {beta}$',clip_on=False,zorder=100)
    
    plt.legend()

    plt.xlim(0,0.004)
    plt.ylim(0.65,1)
    plt.xlabel('Energy-weighted FPR')
    plt.ylabel('Energy-weighted TPR')
    plt.tight_layout()

    plt.savefig('plots/ROCCurve.pdf')

if doSignalPlot:

    for variable in ['Matched-EReco','Matched-ETrue','Unmatched-EReco','Unmatched-ETrue']:
        fig,ax = plt.subplots()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=0))
        for particle in data['particle'].unique():
            dataTemp = data[data['particle'] == particle]
            #plt.plot(dataTemp['beta'],dataTemp[variable],'.',color='k',linestyle=next(lineCycler),label=particle)
            plt.plot(dataTemp['beta'],dataTemp[variable],'.',color=next(colorCyclerPart),linestyle='-',label=particle)

        plt.legend()

        plt.xlim(0,1)
        #plt.ylim(0.65,1)
        plt.xlabel(rf'$\beta$')
        plt.ylabel(labelDict[variable])
        plt.tight_layout()

        plt.savefig(f'plots/{variable}VsBeta.pdf')
        plt.cla()