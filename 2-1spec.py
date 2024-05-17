#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:31:11 2021

@author: lyoung

PURPOSE: compare carma 2-1 and iram 30m 2-1 spectra of 4526.

best way to do this is to take the carma cube, convolve with 30m beam at 2-1 and select central spectrum.


conclusion so far: this is not too bad, and it looks like the 2-1 carma data aren't missing much.
possibly some, but not a lot.
radial trends in 2-1/1-0 would probably be reliable.

05may2021 adding an estimate of the continuum!

TO DO:

"""

import numpy as np
import matplotlib.pyplot as plt


# continuum estimate to be taken out of the carma spectrum since it wasn't done in earlier imaging
contlevel = 0.030 # in Jy

# read spectra from files
carma = np.genfromtxt('12co21_orig.spec.txt', usecols=[0,1], names=['vel','flux'],\
                       comments='#', dtype=None)
iram30m = np.genfromtxt('NGC4526.21.5KMS.txt', usecols=[1,2], names=['vel','tastar'],\
                        comments='#', skip_header=1, dtype=None)
# carma data cube was convolved to match the resolution of the IRAM 30m,
# and then we extracted some individual pixel spectra from that smoothed cube
carmasm_center = np.genfromtxt('12co21_12asec_centerpix.txt', usecols=[0,1], names=['vel','flux'], comments='#', dtype=None)
carmasm_off = np.genfromtxt('12co21_12asec_offset.txt', usecols=[0,1], names=['vel','flux'], comments='#', dtype=None)

# 30m spectra are stored in T_A* units and should be converted to Jy for comparison to carma
# Combes et al say the efficiency at 1.3mm is 0.57
# Atlas3d paper4 says it's 0.63 and 4.73 Jy/K.  I thought it was 4.95 Jy/K but maybe that is older or newer?
iramcorr = (iram30m['tastar']/0.63)*4.73
iramcorr2 = (iram30m['tastar']/0.57)*4.95


# plot figure
plt.figure()
plt.plot(carma['vel'], carma['flux']-contlevel, drawstyle='steps-mid', color='darkgrey', label='CARMA $^{12}$CO 2-1 (full disk)')
plt.plot(iram30m['vel'], iramcorr, drawstyle='steps-mid', color='orange', label='IRAM 30m')
plt.plot(carmasm_off['vel'], carmasm_off['flux']-contlevel, drawstyle='steps-mid', color='blue', label='simulated SD')
plt.plot()
plt.axhline(0.0)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Flux density (Jy)')
textcolors=['grey','orange','blue']
leg = plt.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
for i in enumerate(leg.get_texts()):
     i[1].set_color(textcolors[i[0]])
for item in leg.legendHandles:
            item.set_visible(False)
ax = plt.gca()
ax.axes.tick_params(which='both', direction='in')

plt.savefig('4526_21spec+30m.pdf')
