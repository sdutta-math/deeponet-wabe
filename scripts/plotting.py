#! /usr/bin/env python

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, ScalarFormatter, FormatStrFormatter


import matplotlib.ticker as ticker
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# Plot parameters
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 20,
                     'lines.linewidth': 2,
                     'lines.markersize':10,
                     'axes.labelsize': 16, 
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 16,
                     'axes.linewidth': 2})

import itertools
colors = itertools.cycle(['r','g','b','m','y','c'])
markers = itertools.cycle(['p','d','o','^','s','x',]) #'D','H','v','*'])


def plot_bounds_1d(p1,p2,p3,L,T, label1=None, vmin1=None, vmax1=None, savefig=False, 
                   name=None):
    
    if vmin1 is None:
        vmin1 = np.amin([p1.min(), p2.min()])
        vmax1 = np.amax([p1.max(), p2.max()])

    fig, ((ax1), (ax2), (ax3)) = plt.subplots(1,3, figsize=(20,3))
    pcm1 = ax1.imshow(p1,cmap='jet',origin='lower',
                      vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.79)
    ax1.yaxis.set_ticks(np.arange(0,L+0.1,1))
    fig.colorbar(pcm1,ax=ax1)
    ax1.set_title('Truth' +'\n'+ '%s = %.2f'%('Re', label1)+'\n'+
                 '%.4f<u<%.4f'%(tf.reduce_min(p1).numpy(), tf.reduce_max(p1).numpy()))
    pcm2 = ax2.imshow(p2,cmap='jet',origin='lower',
                      vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.79)
    ax2.yaxis.set_ticks(np.arange(0,L+0.1,1))

    fig.colorbar(pcm2,ax=ax2)
    ax2.set_title('Prediction' +'\n'+ '%s = %.2f'%('Re', label1)+'\n'
                 '%.4f<u<%.4f'%(tf.reduce_min(p2).numpy(), tf.reduce_max(p2).numpy()))
    pcm3 = ax3.imshow(p3,cmap='coolwarm',origin='lower',
                      vmin=-0.05, vmax=0.05, extent=(0,T,0,L), aspect = 0.79)
    ax3.yaxis.set_ticks(np.arange(0,L+0.1,1))

    cbar = fig.colorbar(pcm3,ax=ax3)
    ax3.set_title('Relative Error' +'\n'+ '%s = %.2f'%('Re', label1))
    
    ax1.set_ylabel('$x$',fontsize=18)
    ax1.set_xlabel('$t$',fontsize=18)
    ax2.set_xlabel('$t$',fontsize=18) 
    ax3.set_xlabel('$t$',fontsize=18)
    
    fig.tight_layout()
    if savefig:
        plt.savefig(name)



def plot_spcaetime_1d(p1,p2,p3,p4,p5,p6,T,L, colormap='jet', label1=None, label2=None, 
                      vmin1=None, vmax1=None, savefig=False, name=None):
    """
    Plot space-time 2d plots of 1D solutions
    Row1 : Predicted, True, Error for Soln1
    Row2 : Predicted, True, Error for Soln2
    """
    f = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(3, 3, )
    gs.update(wspace=0.2, hspace=0.2) # set the spacing between axes.

    if vmin1 is None:
        vmin1 = np.amin([p1.min(), p2.min(), p3.min(), p4.min()])
        vmax1 = np.amax([p1.max(), p2.max(), p3.max(), p4.max()])

    ax1 = plt.subplot(gs[0, 0]);
    f1= ax1.imshow(p1,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax1.yaxis.set_ticks(np.arange(0,L+0.1,1)); ax1.xaxis.set_ticks(np.arange(0,T+0.1,1))
    ax1.set_title('%s=%.2f'%('Re',label1[0]))

    ax2 = plt.subplot(gs[0, 1]);
    f2 = ax2.imshow(p2,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax2.yaxis.set_ticks(np.arange(0,L+0.1,1)); ax2.xaxis.set_ticks(np.arange(0,T+0.1,1))
    ax2.set_title('%s=%.2f'%('Re',label1[1]))

    ax3 = plt.subplot(gs[1, 0]);
    f3= ax3.imshow(p3,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax3.set_xticklabels([]); ax3.set_yticklabels([])
#     ax3.yaxis.set_ticks(np.arange(0,1.1,1))
    ax3.set_title('%s=%.2f'%('Re',label1[2]))

    ax4 = plt.subplot(gs[1, 1]);
    f4 = ax4.imshow(p4,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax4.set_xticklabels([]); ax4.set_yticklabels([])
    ax4.set_title('%s=%.2f'%('Re',label1[3]))
    
    ax5 = plt.subplot(gs[2, 0]);
    f5= ax5.imshow(p5,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    ax5.set_xticklabels([]); ax5.set_yticklabels([])
#     ax3.yaxis.set_ticks(np.arange(0,1.1,1))
    ax5.set_title('%s=%.2f'%('Re',label1[4]))

    ax6 = plt.subplot(gs[2, 1]);
    f6 = ax6.imshow(p6,cmap=colormap,origin='lower',vmin=vmin1, vmax=vmax1, extent=(0,T,0,L), aspect = 0.59)
    cbar1 = f.colorbar(f6, ax=list((ax1, ax2, ax3, ax4, ax5, ax6)),orientation='horizontal',aspect=50, pad=0.1)
    ax6.set_xticklabels([]); ax6.set_yticklabels([])
    ax6.set_title('%s=%.2f'%('Re',label1[5]))

    ax1.set_ylabel('$x$',fontsize=18); #ax2.set_ylabel('$x$',fontsize=18);
    ax3.set_ylabel('$x$',fontsize=18); #ax4.set_ylabel('$x$',fontsize=18);

    ax5.set_xlabel('$t$',fontsize=18); ax6.set_xlabel('$t$',fontsize=18); 
    ax5.set_ylabel('$x$',fontsize=18); #ax6.set_ylabel('$x$',fontsize=18);
    if savefig:
        plt.savefig(name)
