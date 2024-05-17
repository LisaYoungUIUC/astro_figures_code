#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:09:22 2020

@author: lyoung

PURPOSE: resolved image of OIII/Hbeta in ngc4526 based on atlas3d data.
      


    
TO DO:

"""

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import os.path
from astropy import wcs
from sauron_colormap import sauron
import aplpy

prefix = 'Dropbox/projects/alma4526/' 
prefix2 = 'projects/atlas3d/opticaldata_release/'

def tedious_repetitive():
    myfig.recenter(racen, deccen, width=width, height=height)
    myfig.ticks.set_color('black')
    myfig.ticks.set_linewidth(1)
    myfig.ticks.set_length(4)
    ax = bigfig.gca()
    ax.tick_params(direction='in')
    ax.set_ylabel('Dec (ICRS)', labelpad=-0.5)
    myfig.add_colorbar()
    myfig.add_scalebar(0.00351)  # I think this is correct for 4526 @ 16.3 Mpc
    myfig.scalebar.set_label('1 kpc')
    ax = bigfig.gca()
    ax.tick_params(direction='in')
    
def tedious2():
    ax = bigfig.gca()
    ax.tick_params(direction='in')

# generic stuff about 4526
racen = 188.512856
deccen = 7.699524
width = 0.013
height = 0.0065

contfile = prefix+'cont_bigmfs_2asec_2t.im.pbcor.fits'  
contheader = pyfits.open(contfile)[0].header
cont_bmaj = contheader['bmaj']
cont_bmin = contheader['bmin']
cont_bpa = contheader['bpa']
cont_levs = 1.25e-5*np.array([-2,2,4,8,16,64,256])  

cofile = prefix+'12co_hires.integrated.fits'
coheader = pyfits.open(cofile)[0].header
co_bmaj = coheader['bmaj']
co_bmin = coheader['bmin']
co_bpa = coheader['bpa']
co_levs = [0.063,0.2,0.4,0.8,1.6,2.5,2.7]  

# after running 4526_dumpfitsfiles.py you can do these
# overlay oiii/hb and radio continuum contours
bigfig = plt.figure()
myfig = aplpy.FITSFigure(prefix+'NGC4526_logOIIIHB.fits', figure=bigfig, auto_refresh=False, north=True)
myfig.show_colorscale(stretch='linear', cmap='viridis_r', vmin=-0.6, vmax=0.6)
tedious_repetitive()
myfig.show_contour(contfile, colors='white', overlap=False, levels=cont_levs, linewidths=1)
myfig.add_beam(major=cont_bmaj, minor=cont_bmin, angle=cont_bpa, color='blue', corner='bottom left', fill=True, linewidth=1)
myfig.colorbar.set_axis_label_text('log([OIII]/H$\\beta$)')
tedious2() # for some reason i think you have to do this after the colorbar axis label call
bigfig.savefig('4526oiiihbcont.pdf')

# OIII/Hb and CO contours
bigfig = plt.figure()
myfig = aplpy.FITSFigure(prefix+'NGC4526_logOIIIHB.fits', figure=bigfig, auto_refresh=False, north=True)
myfig.show_colorscale(stretch='linear', cmap='viridis_r', vmin=-0.6, vmax=0.6)
tedious_repetitive()
myfig.show_contour(cofile, colors='white', overlap=False, levels=co_levs, linewidths=1)
myfig.add_beam(major=co_bmaj, minor=co_bmin, angle=co_bpa, color='blue', corner='bottom left', fill=True, linewidth=1)
myfig.ax.annotate('CO', xy=(0.05,0.85), xycoords='axes fraction')
myfig.colorbar.set_axis_label_text('log([OIII]/H$\\beta$)')
tedious2()
bigfig.savefig('4526oiiihbco.pdf')

# OIII intensity and CO contours
bigfig = plt.figure()
myfig = aplpy.FITSFigure(prefix+'NGC4526_OIII_GAS.fits', figure=bigfig, auto_refresh=False, north=True)
myfig.show_colorscale(stretch='log', vmin=0.005, vmax=0.4, cmap=sauron)
tedious_repetitive()
myfig.show_contour(cofile, colors='white', overlap=False, levels=co_levs, linewidths=1)
myfig.add_beam(major=co_bmaj, minor=co_bmin, angle=co_bpa, color='blue', corner='bottom left', fill=True, linewidth=1)
myfig.colorbar.set_axis_label_text('log [O III] intensity')
tedious2()
bigfig.savefig('4526oiiico.pdf')


# OIII intensity and radio continuum contours
bigfig = plt.figure()
myfig = aplpy.FITSFigure(prefix+'NGC4526_OIII_GAS.fits', figure=bigfig, auto_refresh=False, north=True)
myfig.show_colorscale(stretch='log', vmin=0.005, vmax=0.4, cmap=sauron)
tedious_repetitive()
myfig.show_contour(contfile, colors='white', overlap=False, levels=cont_levs, linewidths=1)
myfig.add_beam(major=cont_bmaj, minor=cont_bmin, angle=cont_bpa, color='blue', corner='bottom left', fill=True, linewidth=1)
myfig.colorbar.set_axis_label_text('log [O III] intensity')
tedious2()
bigfig.savefig('4526oiiicont.pdf')


# Hbeta intensity and radio continuum contours
bigfig = plt.figure()
myfig = aplpy.FITSFigure(prefix+'NGC4526_HB_GAS.fits', figure=bigfig, auto_refresh=False, north=True)
myfig.show_colorscale(stretch='log', vmin=0.005, vmax=0.8, cmap=sauron)
tedious_repetitive()
myfig.show_contour(contfile, colors='white', overlap=False, levels=cont_levs, linewidths=1)
myfig.add_beam(major=cont_bmaj, minor=cont_bmin, angle=cont_bpa, color='blue', corner='bottom left', fill=True, linewidth=1)
myfig.colorbar.set_axis_label_text('log H$\\beta$ intensity')
tedious2()
bigfig.savefig('4526hbcont.pdf')
