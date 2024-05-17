#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:45:03 2020

@author: lisayoung

PURPOSE: plot 12co hi res mom0 for asymmetry discussion

STATUS:

TO DO:
   

    
DONE:
    - try the hi res continuum image on here too, to double-check position of nucleus
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# from astropy import coordinates as coords
from astropy.io import fits
import aplpy
from astropy.wcs import WCS
import cmasher as cmr
import datetime

dolines = False # whether to put on the guide-the-eye straight lines 
docontour = True
dohst = True
# cmap = 'jet'
# cmap = cmr.heat_r  # see for alternatives: https://github.com/1313e/CMasher
cmap = cmr.get_sub_cmap('cmr.heat', 0.15, 1.0)


# setup stuff; coordinates, PAs, inclinations, contour levels etc
racen = 188.5125
deccen = 7.6994222
width = 0.008
height = 0.004
asec2deg = 1./3600.
inc = 77.
pa = np.array([292.,290.,290.])-90.  # the 90 is for conversion to "astro-style" PAs but I'm not actually sure if it's + or -
majax = asec2deg*np.array([29,26,22]) # ellipse semimajor axis?
colevs = [0.05,0.1,0.2,0.4,0.8,1.6]
colevs = [0.05,0.3,0.9]
contlevs = 3.5e-5*np.array([-3,3,10,30,50,95]) # 50sigma is about 1/2 of peak, in this case
linelength = 14./3600. # half-length of line, in degrees
lineoffset = 2.8/3600.  # distance the lines should be offset to one side of nucleus
linepa = 291.-270. # same comment as for ellipse PAs above
# for hand-drawing some straight lines on the image (see below)
blah1 = linelength * np.cos(linepa*np.pi/180.)
blah2 = linelength * np.sin(linepa*np.pi/180.)
blah3 = lineoffset * np.sin(linepa*np.pi/180.)
blah4 = lineoffset * np.cos(linepa*np.pi/180.)
lineendsx_w = np.array([racen+blah1, racen-blah1])
lineendsy_w = np.array([deccen-blah2,deccen+blah2])
lineend1_w = np.array([racen+blah1, deccen-blah2])
lineend2_w = np.array([racen-blah1, deccen+blah2])


#%%
#     start primary figure here

coim = '12co_hires.integrated.fits'
contim = 'cont_hires_spw21_mfs_2.fits'
vmin = 0.0
vmax = 2.8
asecperpix = 0.2 # in my hires 12co images; don't forget to change this if you use different images
if (dohst):
    bigfig = plt.figure(figsize=(7,7))
    fig = aplpy.FITSFigure(coim, figure=bigfig, subplot=[0.2,0.49,0.7,0.39])
else:
    fig = aplpy.FITSFigure(coim)
fig.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
fig.recenter(racen, deccen, width=width, height=height)
if (docontour):
    fig.show_contour(coim, levels=[0.05], colors='w') # 0.4 was 36 Msun/pc2, see text. 
fig.show_markers(racen, deccen, marker='+', facecolor='k', s=100)
fig.set_nan_color('black')
fig.add_beam(corner='bottom right')
fig.beam.set_color('darkgrey')


# add lines - the aplpy native doesn't seem to work, or I can't figure it out
f = fits.open(coim)
w = WCS(f[0].header)
if (dolines):
    lineendsx_p, lineendsy_p = w.all_world2pix(lineendsx_w, lineendsy_w, 0)  # I'm never sure if I want 0-indexing or 1-indexing
    # plt.plot(lineendsx_p, lineendsy_p, color='w') # this line goes through the center
    lineendsx_p, lineendsy_p = w.all_world2pix(lineendsx_w+blah3, lineendsy_w+blah4, 0)
    plt.plot(lineendsx_p, lineendsy_p, color='w', linestyle=':')
    lineendsx_p, lineendsy_p = w.all_world2pix(lineendsx_w-blah3, lineendsy_w-blah4, 0)
    plt.plot(lineendsx_p, lineendsy_p, color='w', linestyle=':')
ax = plt.gca()
ax.tick_params(direction='in', length=8)


if (dohst):
    f2 = aplpy.FITSFigure('4526_jpg_rgbcube_2d.fits', figure=bigfig, subplot=[0.2,0.1,0.7,0.39])
    f2.recenter(racen, deccen, width=width, height=height)
    f2.show_rgb('4526_jpg.png')
    ax = plt.gca()
    ax.tick_params(direction='in', length=8)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    topax = bigfig.add_axes([0.2,0.93,0.7,0.03])
    #bigax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    cb = mpl.colorbar.ColorbarBase(topax, cmap=cmap, norm=norm, orientation='horizontal')
    topax.tick_params(direction='in')
    cb.set_label('Jy bm$^{-1}$ km s$^{-1}$', labelpad=-2)
    bigfig.savefig('4526_12co+hst.pdf', dpi=300)
else:
    fig.add_colorbar(location='top', axis_label_text='Jy bm$^{-1}$ km s$^{-1}$')
    ax = plt.gca()
    ax.tick_params(direction='in')
    if (dolines):
        plt.savefig('4526_12co_lines.pdf')
    else:
        plt.savefig('4526_12co.pdf')

#%%
    
# part 3: try making a comparison image of the nice color HST jpg to see where the CO ring is?
    
f2 = aplpy.FITSFigure('4526_jpg_rgbcube_2d.fits')
f2.recenter(racen, deccen, width=width, height=height)
f2.show_rgb('4526_jpg.png')
    
    
    
#%%
#
# part 2: slice the hires mom0 along the major axis for identifying ring radius 
# this is ridiculously slow with a full 2d interpolation. 
m0data = f[0].data
xx, yy = np.meshgrid(np.arange(m0data.shape[1]), np.arange(m0data.shape[0]))
# x and y pixel coordinates for some sample points along the major axis.
# 30" length sampled at 0.2" just like pixels.  15"/0.2 = 75 pix
racen_p, deccen_p = w.all_world2pix(racen, deccen, 0)
xrelpix = np.arange(m0data.shape[1]) - racen_p
desiredypix = deccen_p + np.tan(linepa*np.pi/180.) * xrelpix # this would be exactly on the line
nearestypix = np.round(desiredypix).astype(int)
nearestxpix = np.round(xrelpix+racen_p).astype(int)
slicepos = xrelpix *asecperpix / np.cos(linepa*np.pi/180.)
slicedata = m0data[nearestypix,nearestxpix]
plt.figure()
plt.plot(slicepos, slicedata)
plt.plot(-slicepos, slicedata)
# compare to the surface brightness profiles made by 4526pvslicefig.py
sb = np.genfromtxt('SBprof_12CO.txt', usecols=[0,1,2], names=['offset','sb','sbunc'])
plt.errorbar(sb['offset'], sb['sb'], yerr=sb['sbunc'])
plt.axvline(0.0)
plt.xlabel('Offset (")')
plt.ylabel('Surface Brightness')

# save data
outfile = open('hires12comom0slice.txt', 'w')
outfile.write("# {:%Y-%b-%d %H:%M:%S} by plothires12com0.py\n".format(datetime.datetime.now()))
outfile.write('# Position   Intensity  \n')
outfile.write('#   (")      (Jy/b*km/s)  \n')
for i in range(len(slicepos)):
    outfile.write('%5.2f %7.3f\n'%(slicepos[i], slicedata[i]))
outfile.close()
