#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:25:43 2019

@author: lyoung

PURPOSE: side-by-side comparisons of the mom0 images of the various lines in 4526.


TO DO:
    - someday try this rather than kludging all the nans into 0s? ax1.patch.set_facecolor(bg_color)
    

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# where to find files & stuff
myDir = 'projects/alma4526/'

cmap = cmr.heat  # see for alternatives: https://github.com/1313e/CMasher
cmap = cmr.get_sub_cmap('cmr.heat', 0.1, 1.0)

def readfile(name):
    file = fits.open(myDir+name)
    file.info()
    im = file[0].data
    header = file[0].header
    naxis1 = header['naxis1']
    naxis2 = header['naxis2']
    crpix1 = header['crpix1']
    crpix2 = header['crpix2']
    crval2 = header['crval2']
    crval1 = header['crval1']
    cdelt1 = header['cdelt1']
    cdelt2 = header['cdelt2']
    cdelt1_asec = cdelt1*3600.
    print(cdelt1_asec)
    cdelt2_asec = cdelt2*3600.
    try:
        bmaj = header['bmaj']*3600.
    except: 
        bmaj, bmin, bpa = [1.0,1.0,0.0]
    else:
        bmin = header['bmin']*3600.
        bpa = header['bpa']
    # compute coordinate values from header info
    pixnums_x = np.array(np.arange(naxis1)+1, dtype='int') # the +1 is for 1-based fits indices
    pixnums_y = np.array(np.arange(naxis2)+1, dtype='int')
    xvals = (pixnums_x-crpix1)*cdelt1 + crval1  # raw coordinates are in degrees
    yvals = (pixnums_y-crpix2)*cdelt2 + crval2  # 
    # subtract position of continuum peak, so that coordinates are relative to it
    # 12:34:02.999  +07.41.57.935 = 
    xvals -= (12.+34/60.+02.999/3600.)*15. # 188.5125
    yvals -= (7.+41/60.+57.935/3600.) # 7.6994222
    # convert from deg to arcsec
    xvals *= 3600
    yvals *= 3600
    im[np.isnan(im)] = 0. # kludge for 12co image with lots of blanks in it
    extent=[-(max(xvals)-cdelt1_asec/2), -(min(xvals)+cdelt1_asec/2), min(yvals)-cdelt2_asec/2, max(yvals)+cdelt2_asec/2] # outside edges of image; funky signs because cdelt1 < 0
    return im, xvals, yvals, extent, bmaj, bmin, bpa

names = ['12co','12co21','13co','c18o','hcn','hco+','hnc','hnco','hnco54','cs','cn32','cn12','ch3oh','empty1']
labels = ['$^{12}$CO(1-0)','$^{12}$CO(2-1)','$^{13}$CO','C$^{18}$O','HCN','HCO$^+$','HNC','HNCO(4-3)','HNCO(5-4)','CS','CN(3/2-1/2)','CN(1/2-1/2)','CH$_3$OH','no line']
# tweaking figure size does adjust the white space between subplots
fig, axs = plt.subplots(5, 3, sharex='all', sharey='all', gridspec_kw={'hspace':0,'wspace':0},\
                        figsize=(7.95,7.9))
for i,ax in enumerate(names):
    # setup info
    name = names[i]
    if (name == '12co'):
        thisfilename = '12co_hires.integrated.fits'
        errfilename = '12co_hires_integrated_err.fits'
    elif ((name == 'cn32') or (name == 'cn12')):
        thisfilename = name+'_na.mom0.fits'
        errfilename = name+'_na.m0err.fits'
    elif (name == '12co21'):
        thisfilename = '12co21_sm_contsub.m0.fits'
        errfilename = '12co21_sm.m0err.fits'
    elif (name == 'hnco54'):
        thisfilename = name+'_clipmask.m0.fits'  # there is also a _masked.m0.fits but it has some contamination
        errfilename = 'c18o_masked.m0err.fits'   # they're so close together, they have the same noise properties
    else:
        thisfilename = name+'_masked.m0.fits'
        errfilename = name+'_masked.m0err.fits'
    # read the file
    im, xvals, yvals, extent, bmaj, bmin, bpa = readfile(thisfilename)
    # plot image
    subax = plt.subplot(5,3,i+1)
    subfig = plt.imshow(im, origin='lower', vmin=np.nanmin(im), vmax=np.nanmax(im), extent=extent,\
               interpolation='nearest', aspect='equal', cmap=cmap)
    plt.xlim(-17,17)
    plt.ylim(-11,9)
    # outline color for each panel
    for spine in subfig.axes.spines.values():
        spine.set_edgecolor('w')    
    # if you want to verify position of peak wrt continuum
    # plt.axhline(0.0, color='white', linestyle=':')
    # plt.axvline(0.0, color='white', linestyle=':')
    beam = Ellipse(xy=[-15.0, +3.0], width=bmin, height=bmaj, angle=-1.0*bpa) # the -1*bpa is a definition issue
    subax.label_outer()
    subax.add_artist(beam)
    beam.set_facecolor('darkgrey') # 'None'
    errim, xvals, yvals, extent, bmaj, bmin, bpa = readfile(errfilename)
    snratio = im/errim
    plt.contour(snratio, origin='lower', levels=[3.0], extent=extent, colors='white', linestyles=':')
    subax.yaxis.set_minor_locator(AutoMinorLocator(2))
    subax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(which='both', direction='in', top=True, right=True, color='white')
    plt.text(-15.0, 5, labels[i], color='white', fontsize=10)
    # small inset colorbar for each panel, showing the numbers
    axins = inset_axes(subax, width="50%", height="6%", loc='lower right')
    cb = plt.colorbar(subfig, cax=axins, orientation="horizontal") #label='km s$^{-1}$') # ticks=[-15,0,15]
    axins.xaxis.set_ticks_position("top")
    cb.outline.set_edgecolor('w')
    cb.ax.xaxis.set_tick_params(color='w', labelcolor='w', labelsize='small')

# remove axes for the last subplot if not using it for data
plt.subplot(5,3,15)
plt.gca().set_visible(False)
# add outer labels
fig.subplots_adjust(wspace=0,hspace=0)
bigax = fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
bigax.set_xlabel('Arcsec', labelpad=0) # Use argument `labelpad` to move label downwards.
bigax.set_ylabel('Arcsec', labelpad=3)
fig.savefig('mom0panels.pdf')