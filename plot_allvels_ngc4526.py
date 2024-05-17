#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:47:21 2020

@author: lyoung

PURPOSE: see if I can make one figure showing the stellar, CO, and ionized gas velocity fields of ngc4526
all at once on the same color scale.

aplpy is not very smart about the redundant 3rd dimension so I have had to strip it off by hand in casa.
(using exportfits with dropdeg=True)

put an actual WCS header on the atlas3d stellar velocity field fits file; that is in 4526_dumpfitsfiles.py


TO DO:



"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
# from astropy import coordinates as coords
from astropy.io import fits
import aplpy
from sauron_colormap import sauron
from astropy import wcs



figysize=10
figxsize=6
aspect=figxsize/figysize
bigfig = plt.figure(figsize=(figxsize,figysize))
vmin, vmax = 230,1000
#cmap = 'jet' 
cmap = sauron

vsys = 615.
racen = 188.512856
deccen = 7.699524
width = 0.009
height = 0.0045
asec2deg = 1./3600.
inc = 77.

vellevs = np.array([-335,-300,-250,-200,-150,-100,-50,0,50,100,150,200,250,300,335]) + vsys

# parameters for the majax and minax lines
nuc_ra = 188.5125159
nuc_dec = 7.6994095
linelength = 0.0111 # degrees
majaxpa_d = 293.0-90.
majaxpa = majaxpa_d * np.pi/180.  # the 90 here is to get the astronomy definition wrt North
minaxpa = majaxpa - np.pi/2.
trial_minaxpa = minaxpa + (12.*np.pi/180.) # offset to describe how much the vsys contour is rotated from perp
majax_x = [(nuc_ra + 0.5*linelength*np.cos(majaxpa)), (nuc_ra - 0.5*linelength*np.cos(majaxpa))]
majax_y = [(nuc_dec - 0.5*linelength*np.sin(majaxpa)), (nuc_dec + 0.5*linelength*np.sin(majaxpa))]
minax_x = [(nuc_ra + 0.5*linelength*np.cos(minaxpa)), (nuc_ra - 0.5*linelength*np.cos(minaxpa))]
minax_y = [(nuc_dec - 0.5*linelength*np.sin(minaxpa)), (nuc_dec + 0.5*linelength*np.sin(minaxpa))]
trial_minax_x = [(nuc_ra + 0.5*linelength*np.cos(trial_minaxpa)), (nuc_ra - 0.5*linelength*np.cos(trial_minaxpa))]
trial_minax_y = [(nuc_dec - 0.5*linelength*np.sin(trial_minaxpa)), (nuc_dec + 0.5*linelength*np.sin(trial_minaxpa))]


#%%

# 3-panel figure with stellar, CO, and ionized gas velocity fields

# panel 1: stars
im = 'NGC4526_VPXF.fits'  # "real" release data dumped to fits file by dumpfitsfiles.py
f3 = aplpy.FITSFigure(im, figure=bigfig, subplot=[0.2,0.63,0.6,0.25], auto_refresh=False)
f3.recenter(racen, deccen, width=width, height=height)
f3.show_colorscale(vmin=vmin, vmax=vmax, cmap=cmap)
f3.show_contour(im, levels=vellevs, colors='k', linewidths=1)
# the 12co contour here is supposed to show kinda the max extent of the molecular gas
f3.show_contour('12co_hires.integrated.fits', levels=[0.1], colors='w', linewidths=1)
f3.axis_labels.set_ypad(-0.5)
hdr = fits.open(im)[0].header
imwcs = wcs.WCS(hdr)
majax_x_pix, majax_y_pix = imwcs.wcs_world2pix(majax_x, majax_y, 0)
minax_x_pix, minax_y_pix = imwcs.wcs_world2pix(minax_x, minax_y, 0)
plt.plot(majax_x_pix, majax_y_pix, linestyle=':', color='w')
plt.plot(minax_x_pix, minax_y_pix, linestyle=':', color='w')
f3.axis_labels.hide_x()
f3.tick_labels.hide_x()
ax = bigfig.gca()
ax.tick_params(direction='in', color='k')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.annotate('stars', xy=(0.08,0.9), xycoords='axes fraction')


# panel 2: CO
im = 'gausshermitefits/fitpeakvels.fits' 
f2 = aplpy.FITSFigure(im, figure=bigfig, subplot=[0.2,0.38,0.6,0.25], auto_refresh=False)
f2.recenter(racen, deccen, width=width, height=height)
f2.show_colorscale(vmin=vmin, vmax=vmax, cmap=cmap)
f2.show_contour(im, levels=vellevs, colors='darkgrey', linewidths=1)
f2.axis_labels.set_ypad(-0.5)
hdr = fits.open(im)[0].header
imwcs = wcs.WCS(hdr)
majax_x_pix, majax_y_pix = imwcs.wcs_world2pix(majax_x, majax_y, 0)
minax_x_pix, minax_y_pix = imwcs.wcs_world2pix(minax_x, minax_y, 0)
plt.plot(majax_x_pix, majax_y_pix, linestyle=':', color='w')
plt.plot(minax_x_pix, minax_y_pix, linestyle=':', color='k')
blahx, blahy = imwcs.wcs_world2pix(racen-4./3600., deccen-4./3600., 0)
plt.text(blahx, blahy, 'far - receding')
blahx, blahy = imwcs.wcs_world2pix(racen+9./3600., deccen-7.5/3600., 0)
plt.text(blahx, blahy, 'far - approaching')
blahx, blahy = imwcs.wcs_world2pix(racen+10.5/3600., deccen+2./3600., 0)
plt.text(blahx, blahy, 'near-\napproaching')
blahx, blahy = imwcs.wcs_world2pix(racen-2.5/3600., deccen+5.5/3600., 0)
plt.text(blahx, blahy, 'near - receding')
f2.add_beam(corner='bottom right')
f2.beam.set_color('darkgrey')
f2.axis_labels.hide_x()
f2.tick_labels.hide_x()
ax = bigfig.gca()
ax.tick_params(direction='in', color='k')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.annotate('CO', xy=(0.05,0.9), xycoords='axes fraction')



# panel 3: ionized gas vels
im = 'NGC4526_V_OIII_clip.fits'
myfig = aplpy.FITSFigure(im, figure=bigfig, subplot=[0.2,0.13,0.6,0.25], auto_refresh=False)
myfig.recenter(racen, deccen, width=width, height=height)
myfig.show_colorscale(vmin=vmin, vmax=vmax, cmap=cmap)
myfig.show_contour(im, levels=vellevs, colors='k', linewidths=1)
myfig.axis_labels.set_ypad(-0.5)
hdr = fits.open(im)[0].header
imwcs = wcs.WCS(hdr)
majax_x_pix, majax_y_pix = imwcs.wcs_world2pix(majax_x, majax_y, 0)
minax_x_pix, minax_y_pix = imwcs.wcs_world2pix(minax_x, minax_y, 0)
plt.plot(majax_x_pix, majax_y_pix, linestyle=':', color='w')
plt.plot(minax_x_pix, minax_y_pix, linestyle=':', color='k')
myfig.show_contour('12co_hires.integrated.fits', levels=[0.1], colors='w', linewidths=1)
ax = bigfig.gca()
ax.tick_params(direction='in', color='k')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.annotate('[O III]', xy=(0.05,0.9), xycoords='axes fraction')


# wrap up this figure
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
bigax = bigfig.add_axes([0.2,0.93,0.6,0.03])
cb = mpl.colorbar.ColorbarBase(bigax, cmap=cmap, norm=norm, orientation='horizontal')
bigax.tick_params(direction='in')
cb.set_label('Velocity (km s$^{-1}$)', labelpad=-2)
bigfig.savefig('ngc4526allvels.pdf')


#%%

# big one for estimating angle between vsys and perp

im = 'gausshermitefits/fitpeakvels.fits' 
f4 = aplpy.FITSFigure(im, auto_refresh=False)
f4.recenter(racen, deccen, width=width, height=height)
f4.show_colorscale(vmin=vmin, vmax=vmax, cmap=cmap)
f4.show_contour(im, levels=vellevs, colors='darkgrey', linewidths=1)
# here I am preparing to add some lines on the velocity field.
# lines to mark the kinematic major axis and minor axis, along with some 
# trials since the kinematic minor axis is not exactly perpendicular to major axis
hdr = fits.open(im)[0].header
imwcs = wcs.WCS(hdr)
majax_x_pix, majax_y_pix = imwcs.wcs_world2pix(majax_x, majax_y, 0)
minax_x_pix, minax_y_pix = imwcs.wcs_world2pix(minax_x, minax_y, 0)
trial_minax_x_pix, trial_minax_y_pix = imwcs.wcs_world2pix(trial_minax_x, trial_minax_y, 0)
plt.plot(majax_x_pix, majax_y_pix, linestyle=':', color='grey')
plt.plot(minax_x_pix, minax_y_pix, linestyle=':', color='grey')
plt.plot(trial_minax_x_pix, trial_minax_y_pix, linestyle=':', color='w')
blahx, blahy = imwcs.wcs_world2pix(racen-5./3600., deccen-3./3600., 0)
plt.text(blahx, blahy, 'far side')


#%%

# CO by itself

im = 'gausshermitefits/fitpeakvels.fits' 
justco = plt.figure('justco')
f5 = aplpy.FITSFigure(im, figure=justco, auto_refresh=False)
f5.recenter(nuc_ra, nuc_dec, width=0.008, height=0.004)
f5.show_colorscale(vmin=vmin, vmax=vmax, cmap=cmap)
f5.show_contour(im, levels=vellevs, colors='darkgrey', linewidths=1)
hdr = fits.open(im)[0].header
imwcs = wcs.WCS(hdr)
majax_x_pix, majax_y_pix = imwcs.wcs_world2pix(majax_x, majax_y, 0)
minax_x_pix, minax_y_pix = imwcs.wcs_world2pix(minax_x, minax_y, 0)
plt.plot(majax_x_pix, majax_y_pix, linestyle=':', color='grey')
plt.plot(minax_x_pix, minax_y_pix, linestyle=':', color='grey')
f5.add_beam(corner='bottom right')
f5.beam.set_color('darkgrey')
f5.axis_labels.set_ypad(-0.4)
ax = justco.gca()
ax.tick_params(direction='in', color='k')
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
f5.add_colorbar()
f5.axis_labels.set_ypad(0)
f5.colorbar.set_location('top')
f5.colorbar.set_axis_label_text('Velocity (km s$^{-1}$)')
ax = justco.gca()
ax.tick_params(direction='in', color='k')
f5.savefig('ngc4526covels.pdf', dpi=300) # specifying a higher dpi fixes a weird error with the colorbar and its outline
f5.savefig('ngc4526covels.png')