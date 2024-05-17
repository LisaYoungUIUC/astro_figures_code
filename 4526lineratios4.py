#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:43:55 2019

@author: lyoung

PURPOSE: given two data cubes, form spectra from various annular regions on the sky
  and then do best fit line ratios.  
  You have to run it once for each line, specifying the line by hand.
  
  This version is copied from 4526lineratios2.py on 21feb2021 and modified for use with the 
  new 12co data at 2" res, and the matching 13co and c18o cubes at that resolution.

  
TO DO:

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy import wcs
import datetime

# definitions
restfreq_hcn = 88.631601 # GHz
restfreq_13co = 110.2013543
restfreq_hcop = 89.188526
restfreq_c18o = 109.7821734
restfreq_12co = 115.2712018
restfreq_hnc = 90.663564
restfreq_cs = 97.98095330
restfreq_hnco = 87.92523700
restfreq_ch3oh = 96.741375  # approx, probably good enough for this
restfreq_cn32 = 113.49097
restfreq_cn12 = 113.17049150 # this is a bit uncertain
restfreq_12co21 = 230.53800000

# This is the function that calculates the "model" for the fitting routine.  It is scaled version of the cube1 spec.
def scaled_spec1(channels, factor):
    # linechans are defined in the main routine and they are inherited
    return spec1[linechans]/factor

# other generic setup
myDir = 'projects/alma4526/'

# adopted disk parameters for ngc4526
inc = 78 # disk inclination in degrees
theta = -23*np.pi/180  # rotation angle in radians
centerra = 15.0 * (12.0 + 34.0/60 + 02.997/3600) # 12h34m02.997s converted to degrees
centerdec = 7.0 + 41.0/60 + 57.87/3600 # 7d41m57.87s converted to decimal degrees
ringboundaries = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0])  # semimajor axes, in arcsec
ringboundaries_deg = ringboundaries/3600.
nrings = len(ringboundaries) - 1

# read first data cube and get WCS info from its header.
line1name = '12CO21contsub' # choose from 12co, 13co, etc. note 12co21 can be either "as is" or contsub.
if (line1name == '12CO'):
    cube1file = fits.open(myDir+'12co_2asec_15kms.image.pbcor.fits')
    line1restfreq = restfreq_12co
    startchan = 15  # defines the ends of the channel range where the line lives
    stopchan = 62 # if you want this to be in the line, use the +1 below in the linechans[]
if (line1name == '13CO'):
    cube1file = fits.open(myDir+'13co_15kms_morechans.image.pbcor.fits')
    line1restfreq = restfreq_13co
    startchan = 15  # defines the ends of the channel range where the line lives
    stopchan = 62 # if you want this to be in the line, use the +1 below in the linechans[]
if (line1name == 'CN32'):
    cube1file = fits.open(myDir+'cn32_2asec.15kms.image.pbcor.fits')
    line1restfreq = restfreq_cn32
    startchan = 15  # defines the ends of the channel range where the line lives
    stopchan = 62 # if you want this to be in the line, use the +1 below in the linechans[]
if (line1name == '12CO21'):
    cube1file = fits.open(myDir+'12co21_sm.fits')  # basically you should only use this cube with 12co_hires.im.pb.sm.rgrd.fits
    line1restfreq = restfreq_12co21
    startchan = 8  # defines the ends of the channel range where the line lives
    stopchan = 79 # if you want this to be in the line, use the +1 below in the linechans[]
    # this particular set, the 2-1/1-0 line ratio, can be done at even higher res.  1.12" beam.
    ringboundaries = np.array([0.0, 0.55, 1.65, 2.75, 3.85, 4.95, 6.05, 7.15, 8.25, 9.35, 10.45, 11.55, 15]) # in arcsec; outer edges of regions to use for measurements
    ringboundaries_deg = ringboundaries/3600.
    nrings = len(ringboundaries) - 1
if (line1name == '12CO21contsub'):
    cube1file = fits.open(myDir+'12co21_sm_contsub.fits')  # basically you should only use this cube with 12co_hires.im.pb.sm.rgrd.fits
    line1restfreq = restfreq_12co21
    startchan = 8  # defines the ends of the channel range where the line lives
    stopchan = 79 # if you want this to be in the line, use the +1 below in the linechans[]
    # this particular set, the 2-1/1-0 line ratio, can be done at even higher res.  1.12" beam.
    ringboundaries = np.array([0.0, 0.55, 1.65, 2.75, 3.85, 4.95, 6.05, 7.15, 8.25, 9.35, 10.45, 11.55, 15]) # in arcsec; outer edges of regions to use for measurements
    ringboundaries_deg = ringboundaries/3600.
    nrings = len(ringboundaries) - 1
cube1 = cube1file[0].data  # python refers to the axes as channel, ??, ??
cube1file.info()
w = wcs.WCS(cube1file[0].header)
channels = np.arange(cube1.shape[0]) # this will be zero-based
linechans = np.full(cube1.shape[0], False)
linechans[startchan:stopchan+1] = True
linefreechans = np.logical_not(linechans)
nlinechans = np.sum(linechans)

# read second data cube.  Assuming its WCS is the same as the first cube! (except for frequency)
line2name = '13CO' # choose from c18o, 13co, etc
if (line2name == 'C18O'):
    cube2file = fits.open(myDir+'c18o_15kms_morechans.image.pbcor.fits')
    line2restfreq = restfreq_c18o
if (line2name == '13CO'):
    cube2file = fits.open(myDir+'13co_15kms_morechans.image.pbcor.fits')
    line2restfreq = restfreq_13co
if (line2name == 'CN32'):
    cube2file = fits.open(myDir+'cn32_2asec.15kms.image.pbcor.fits')
    line2restfreq = restfreq_cn32
if (line2name == 'CN12'):
    cube2file = fits.open(myDir+'cn12_2asec.15kms.image.pbcor.fits')
    line2restfreq = restfreq_cn12
if ((line1name == '12CO21') or (line1name == '12CO21contsub')): # yes that's line1, special deal for 2-1/1-0 ratio
    line2name = '12CO'
    cube2file = fits.open(myDir+'12co_hires_round.im.pb.rgrd.fits')  
    line2restfreq = restfreq_12co
#

#
cube2 = cube2file[0].data
w2 = wcs.WCS(cube2file[0].header)
assert cube1.shape == cube2.shape, 'Are these the same region?'

# constructing the mask -- background stuff that is the same for all annuli
xvals_pix = np.arange(cube1.shape[2])  
yvals_pix = np.arange(cube1.shape[1])  
x_mesh_pix, y_mesh_pix = np.meshgrid(xvals_pix, yvals_pix, sparse=False)
x_mesh_world, y_mesh_world, freq_mesh_world = w.wcs_pix2world(x_mesh_pix, y_mesh_pix, 0.0*x_mesh_pix, 0)
x_mesh_rel = x_mesh_world - centerra  # now in degrees relative to chosen center
y_mesh_rel = y_mesh_world - centerdec
costh = np.cos(theta)
sinth = np.sin(theta)
x2_mesh_rot = (costh**2 * x_mesh_rel**2) + (sinth**2 * y_mesh_rel**2) + (2 * x_mesh_rel * y_mesh_rel * costh * sinth)
y2_mesh_rot = (sinth**2 * x_mesh_rel**2) + (costh**2 * y_mesh_rel**2) - (2 * x_mesh_rel * y_mesh_rel * costh * sinth)
dist2 = x2_mesh_rot + (y2_mesh_rot/np.cos(inc*np.pi/180))  # usage: < sma**2 is inside the ellipse

# open the file for output
outfile = open(myDir+'annuli_'+line1name+'_'+line2name+'.txt', 'w')
outfile.write("# {:%Y-%b-%d %H:%M:%S} by 4526lineratios4.py\n".format(datetime.datetime.now()))
outfile.write('# Using inclination = %3.1f\n'%inc)
outfile.write('# Inner  Outer   Ratio  Err \n')

              
# loop over annuli
for i in range(nrings):
    
    a1 = ringboundaries_deg[i]
    a2 = ringboundaries_deg[i+1]
    useme = (dist2 >= a1**2)*(dist2 < a2**2)
    mask = np.zeros(x2_mesh_rot.shape)
    mask[useme] = 1.0
    
    # here's the sum, finally
    spec1 = np.nansum(cube1*mask, axis=(1,2))
    spec2 = np.nansum(cube2*mask, axis=(1,2))
    # cheesy rms estimate just using the end channels (there aren't very many but maybe ok)
    rms2 = np.std(spec2[linefreechans])
    rms1 = np.std(spec1[linefreechans])
    # error on the line2 datapoints.  Here we are assuming line1 is a perfect model with no uncertainty, which is more or less OK when it has high S/N
    # but this does mean you might get different results when fitting line1/line2 than line2/line1, and that may not be a surprise
    rms_arr = 0*spec2 + np.sqrt(rms2**2 + rms1**2)  # will be dominated by whichever is larger, in case they are different.  edited 27apr2021
    # sanity check: is line detected in this annulus?
    line1sum = np.sum(spec1[linechans]) # this is not yet a proper line flux because I don't have channel width
    line2sum = np.sum(spec2[linechans]) # units are probably Jy*chan?
    line1unc = rms1 * np.sqrt(nlinechans) # similarly proper units would require channel width
    line2unc = rms2 * np.sqrt(nlinechans)
    plt.figure('ring '+str(i))
    plt.plot(channels, spec1, drawstyle='steps-mid', label='%.3f +- %.3f S/N=%.1f'%(line1sum,line1unc,(line1sum/line1unc)))
    plt.plot(channels, spec2, drawstyle='steps-mid', label='%.3f +- %.3f S/N=%.1f'%(line2sum,line2unc,(line2sum/line2unc)))
    plt.axhline(0.0, color='k')
    plt.axvspan((startchan-0.5), (stopchan+0.5), color='grey', alpha=0.3)
    plt.legend()
    plt.xlabel('Channel')
    plt.ylabel('Sum')
    
    # sanity checking the selected region is where I want it
    mom0 = np.sum(cube1,axis=0)
    plt.figure('ring '+str(i)+' layout')
    plt.imshow(mom0, origin='low', interpolation='nearest')
    plt.imshow(mask, origin='low', interpolation='nearest', alpha=0.3)
    plt.xlabel('RA or Dec??')
    plt.ylabel('the other one')
    
    # fit spec2 as spec1 * constant.
    popt, pcov = curve_fit(scaled_spec1, channels[linechans], spec2[linechans], sigma=rms_arr[linechans], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    ratio = popt[0]*(line2restfreq/line1restfreq)**2# convert me for K units instead of Jy
    ratioerr = perr[0]*(line2restfreq/line1restfreq)**2 # same conversion factor
    print(line1name,'/',line2name,ringboundaries[i],'to',ringboundaries[i+1],'arcsec, ratio is %.2f'%ratio,'+- %.2f'%ratioerr)
    # write fit results to an output file for safekeeping
    if ((line1sum/line1unc >= 3.0) and (line2sum/line2unc >= 3.0)):  
        outfile.write('%5.1f %5.1f %6.2f %6.2f\n'%(ringboundaries[i], ringboundaries[i+1], ratio, ratioerr))
    # sanity check: recreate scaled spectrum for eyeballing whether the fit is reasonable
    rescaledspec1 = spec1/popt[0] # since these are done on the original data they use the ratio in Jy units
    resid = spec2 - rescaledspec1
    rescaledspec_up = spec1/(popt[0]-perr[0])  # for comparison - do errors look reasonable?
    rescaledspec_dn = spec1/(popt[0]+perr[0])
    #
    plt.figure('ring '+str(i)+' scaled')
    plt.axhline(0.0, color='k')
    plt.plot(channels, rescaledspec1, 'r', drawstyle='steps-mid', label=line1name+'/%.1f'%ratio) # quoted scale factor here is the K units
    plt.plot(channels, spec2, 'k', drawstyle='steps-mid', label='spec2')
    plt.plot(channels, resid, drawstyle='steps-mid', label='resid')
    plt.plot(channels, rescaledspec_up, 'r:', drawstyle='steps-mid')
    plt.plot(channels, rescaledspec_dn, 'r:', drawstyle='steps-mid')
    plt.axvspan((startchan-0.5), (stopchan+0.5), color='grey', alpha=0.3)
    plt.xlabel('Channels')
    plt.ylabel('Sum')  # check units
    plt.legend()
    

outfile.close()