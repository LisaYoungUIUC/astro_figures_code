#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:34:43 2019

@author: lisayoung

PURPOSE: read the data files written by 4526lineratios{2,3}.py and 4526pvslicefig.py.
  Plot them in nice plots showing radial variation of line ratios.
  

TO DO:


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator

# where to find files & stuff
myDir = 'Dropbox/projects/alma4526/'

dosbprof = True # overlay 12co surface brightness profile from majax slice, just for context
dovline = True # marking location of the peak SB on the ring
vloc = 5.2 # arcsec, for the line mentioned above


def tedious_repetitive():
    ax_kpc = ax_asec.twin(aux_trans)
    ax_kpc.set_viewlim_mode("transform")
    ax_asec.axis['bottom'].set_label('Radius (")')
    ax_asec.axis['left'].set_label('Ratio')
    ax_asec.axis['left'].label.set_visible(True)
    ax_kpc.axis["top"].set_label('Radius (kpc)')
    ax_kpc.axis["top"].label.set_visible(True)
    ax_kpc.axes.tick_params(which='both', direction='in')
    ax_asec.axis["right"].major_ticklabels.set_visible(False)
    ax_kpc.axis["right"].major_ticklabels.set_visible(False)
    ax_asec.set_yscale('log')
    ax_asec.get_yaxis().set_ticks_position('both')  # I think this is the important one that makes the ticks the same on both sides
    ax_asec.axes.tick_params(which='both', direction='in')
    ax_asec.get_yaxis().set_tick_params(which='major', length=8)
    y_major = LogLocator(base = 10.0, numticks = 5)
    ax_asec.get_yaxis().set_major_locator(y_major)
    ax_kpc.get_yaxis().set_major_locator(y_major)
    y_minor = LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax_asec.get_yaxis().set_minor_locator(y_minor)
    ax_kpc.get_yaxis().set_minor_locator(y_minor)
    ax_asec.legend(loc='best', frameon=False, handlelength=0)
    plt.xlim(-0.5,17)



def plotme(filename, label, color, dooffset):
    if 'annuli' in filename:
        annuli = np.genfromtxt(filename, usecols=[0,1,2,3], names=['inner','outer','ratio','err'], \
                       dtype=None)
        offsets = np.zeros(annuli.size)
        if (dooffset):
            offsets += 0.5 * (1. - np.random.random_sample(annuli.size))
        xvals = (annuli['inner']+annuli['outer'])/2. + offsets
        ax_asec.errorbar(xvals, annuli['ratio'], yerr=annuli['err'], xerr=(annuli['outer']-annuli['inner'])/2.,\
             fmt='o', color=color, markerfacecolor='None', label=label)
    else:
        pvs = np.genfromtxt(filename, usecols=[0,1,2,3], names=['mid','width','ratio','err'], dtype=None)
        offsets = np.zeros(pvs.size)
        if 'slant' in filename:
            fmt = '^'
        else:
            fmt = 'o'
        if (dooffset):
            offsets += 0.5 * (1. - np.random.random_sample(pvs.size))
        xvals = pvs['mid'] + offsets
        ax_asec.errorbar(xvals, pvs['ratio'], yerr=pvs['err'], xerr=pvs['width'], fmt=fmt, color=color, label=label)
        

# linear scale parameters
asecperkpc = 1000./(4.85*16.4) # 16.4 Mpc for 4526
aux_trans = mtransforms.Affine2D().scale(asecperkpc, 1.)

# fig 1: co isotopes
fig1 = plt.figure()
ax_asec = SubplotHost(fig1, 1, 1, 1, aspect='auto')
fig1.add_subplot(ax_asec)
# 13co/c18o
plotme(myDir+'annuli_13CO_C18O.txt', '', 'forestgreen', False)
plotme(myDir+'pvratios_13CO_C18O.txt', '13CO/C18O', 'forestgreen', False)
# 12co/13co
plotme(myDir+'annuli_12CO_13CO.txt', '', 'blue', False)
plotme(myDir+'pvratios_12CO_13CO.txt', '12CO/13CO', 'blue', False)
# 12co/c18o
plotme(myDir+'pvratios_12CO_C18O.txt', '12CO/C18O', 'red', False)
plotme(myDir+'annuli_12CO_C18O.txt', '', 'red', False)
#
tedious_repetitive()
ax_asec.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
plt.ylim(3.0,40)
fig1.savefig('radial_cos.pdf')



# fig 2:  small ratios
fig2 = plt.figure()
ax_asec = SubplotHost(fig2, 1, 1, 1, aspect='auto')
fig2.add_subplot(ax_asec)
# hcn/hco+
plotme(myDir+'pvratios_HCN_HCO+.txt', 'HCN/HCO+', 'red', False) # 0 (for controlling the legend)
plotme(myDir+'annuli_HCN_HCO+.txt', '', 'red', False)   # 1
# hcn/cs
plotme(myDir+'pvratios_HCN_CS.txt', 'HCN/CS', 'blue', True) # 2
plotme(myDir+'annuli_HCN_CS.txt', '', 'blue', True)  # 3
# ch3oh/hcn
plotme(myDir+'pvratios_CH3OH_HCN.txt', 'CH3OH/HCN', 'grey', False)  # 4
# ch3oh/hnco
plotme(myDir+'pvratios_CH3OH_HNCO.txt', 'CH3OH/HNCO', 'orange', False)  # 5
# cs/cn32
plotme(myDir+'pvratios_CS_CN32.txt', 'CS/CN', 'mediumorchid', False)  # 6
plotme(myDir+'annuli_CS_CN32.txt', '', 'mediumorchid', False)  # 7
# cn32/hcn
plotme(myDir+'pvratios_CN32_HCN.txt', 'CN/HCN', 'black', False)  # 8
plotme(myDir+'annuli_CN32_HCN.txt', '', 'black', False)  # 9
# hnco54/43
plotme(myDir+'pvratios_HNCO54_43.txt', 'HNCO54/43', 'green', True)  # 10

plt.ylim(0.06,15)
tedious_repetitive()
plt.xlim(-0.5,13)
ax_asec.legend(loc=2, frameon=False, handlelength=0) # this one needs a little manual help with legend placement
lines = ax_asec.get_lines()
legend1 = plt.legend([lines[i] for i in [0,2,4]], ["HCN/HCO$^+$", "HCN/CS", "CH$_3$OH/HCN"], loc=2, frameon=False, handlelength=0)
legend2 = plt.legend([lines[i] for i in [5,6]], ['CH$_3$OH/HNCO43','CS/CN32'], loc=1, frameon=False, handlelength=0)
legend3 = plt.legend([lines[i] for i in [8,10]], ['CN32/HCN','HNCO54/43'], loc=9, frameon=False, handlelength=0)
ax_asec.add_artist(legend1)
ax_asec.add_artist(legend2)
ax_asec.add_artist(legend3)
fig2.savefig('radial_bright.pdf')



# fig 3: large ratios.
fig3 = plt.figure()
ax_asec = SubplotHost(fig3, 1, 1, 1, aspect='auto')
fig3.add_subplot(ax_asec)
# 12co/cs
plotme(myDir+'pvratios_12CO_CS.txt', '12CO/CS', 'magenta', False)
plotme(myDir+'annuli_12CO_CS.txt', '', 'magenta', False)
# 12co/ch3oh
plotme(myDir+'pvratios_12CO_CH3OH.txt', '12CO/CH3OH', 'forestgreen', True)
plotme(myDir+'annuli_12CO_CH3OH.txt', '', 'forestgreen', True)
# 12co/hcn
plotme(myDir+'pvratios_12CO_HCN.txt', '12CO/HCN', 'orange', False)
plotme(myDir+'annuli_12CO_HCN.txt', '', 'orange', False)
# hcn/hnco
plotme(myDir+'pvratios_HCN_HNCO.txt', 'HCN/HNCO', 'skyblue', True)
# 12co/hnco
plotme(myDir+'pvratios_12CO_HNCO.txt', '12CO/HNCO', 'blue', True)
plotme(myDir+'annuli_12CO_HNCO.txt', '', 'blue', False)
# 12co/cn32
plotme(myDir+'pvratios_12CO_CN32.txt', '12CO/CN', 'black', True)
plotme(myDir+'annuli_12CO_CN32.txt', '', 'black', False)
if (dosbprof):
        sbprof = np.genfromtxt('hires12comom0slice.txt', usecols=[0,1], names=['off','sb'])
        scalefactor = 180./np.nanmax(sbprof['sb'])
        plt.plot(sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey')
        plt.plot(-sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey') # arbitrary scale factor to get the lines to show up on the plot
if (dovline):
        plt.axvline(vloc, color='k', linestyle=':')
tedious_repetitive()
plt.ylim(5,200)
fig3.savefig('radial_faint.pdf')




# fig 4: more smallish ratios
fig4 = plt.figure()
ax_asec = SubplotHost(fig4, 1, 1, 1, aspect='auto')
fig4.add_subplot(ax_asec)
# 12co/cn
plotme(myDir+'pvratios_12CO_CN32.txt', '12CO/CN32', 'mediumorchid', False)
plotme(myDir+'annuli_12CO_CN32.txt', '', 'mediumorchid', False)
# cn32/cn12
plotme(myDir+'pvratios_CN32_CN12.txt', 'CN32/CN12', 'yellowgreen', False)
plotme(myDir+'annuli_CN32_CN12.txt', '', 'yellowgreen', False)
# cn32/CS and CN/HCN would be good, but I don't think I have the files yet
# hnco54/43
plotme(myDir+'pvratios_HNCO54_43.txt', 'HNCO54/43', 'red', False)
plt.ylim(0.06,40)
tedious_repetitive()
plt.xlim(-0.5,13)
fig4.savefig('radial_yetmore.pdf')