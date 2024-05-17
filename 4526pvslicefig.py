#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:35:20 2019

@author: lyoung

PURPOSE: compute radial variations in line ratios using regions in the PV slices.
write their results to text files for plotting.  Experiments also with estimating uncertainties
in the line ratios using a couple of different methods.  One is a standard kind of statistical
method based on knowing the thermal noise levels of the pixels that are being summed.
Another looks at how the sums change if you raise or lower the clip level that determines
which pixels contribute to the sum.

Notes to myself on clipping/masking/noise calcs.
pvrms1 and pvrms2 are thermal noise levels in the line-free corners of the pv image and they are used for 
estimating the statistical uncertainty in the sums (which get propagated into the uncertainties in the ratios).
For pixel selection I am first applying a mask which is just the clean mask for the cube with higher S/N.
Then there's a clip mask on the brightness of the higher S/N line.  The combination allows the clip level to be
kinda low but not to pick up random noise pixels in the line-free regions.
When I didn't have good 12co data I used the 13co masks and clips for everything, but now that I have 
good 12co data I will probably use the 12co as clip and mask for the 12co/X ratios.


Also the calcratios subroutine returns surface brightness profiles in Jy/b*channel.  
you can spit those out to a file if desired and run the column density calculations on them.

TO DO
- would make this more robust to get asecperpix and fwhm out of the image headers rather than having to 
code them in by hand.
- also would be nicer to loopify a bunch of the repetitive stuff at the end.  
   would probably require a data file with all the not-quite-repetitive bits

   


"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime
from matplotlib.colors import SymLogNorm

# where to find files & stuff
myDir = 'Dropbox/projects/alma4526/'

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
restfreq_hnco54 = 109.9057490

def readfile(name):
    # this version for files that have a frequency axis rather than velocity
    pvfile = fits.open(myDir+name)
    # pvfile.info()
    pvim = pvfile[0].data
    pvheader = pvfile[0].header
    naxis1 = pvheader['naxis1']
    naxis2 = pvheader['naxis2']
    crpix1 = pvheader['crpix1']
    crpix2 = pvheader['altrpix']
    crval2 = pvheader['altrval']
    crval1 = pvheader['crval1']
    cdelt1 = pvheader['cdelt1']
    cdelt2 = -1.*(pvheader['cdelt2']/pvheader['restfrq'])*3e8 # I think this is OK, but should check
    xvals = ((np.arange(naxis1)+1)-crpix1)*cdelt1 + crval1  # these will be in arcsec
    yvals = ((np.arange(naxis2)+1)-crpix2)*cdelt2 + crval2  # these in m/s
    yvals = yvals/1000. # m/s to km/s
    cdelt2 = cdelt2/1000. # likewise
    extent=[xvals[0]-cdelt1/2, xvals[-1]+cdelt1/2, yvals[0]-cdelt2/2, yvals[-1]+cdelt2/2] # outside edges of image
    chanwid_kms = np.abs(cdelt2)
    print(name,'Channel width kms',chanwid_kms)
    return pvim, xvals, yvals, cdelt1, chanwid_kms, extent

def readfilevel(name):
    # this version for files that have a velocity axis rather than frequency
    pvfile = fits.open(myDir+name)
    # pvfile.info()
    pvim = pvfile[0].data
    pvheader = pvfile[0].header
    naxis1 = pvheader['naxis1']
    naxis2 = pvheader['naxis2']
    crpix1 = pvheader['crpix1']
    crpix2 = pvheader['crpix2']
    crval2 = pvheader['crval2']
    crval1 = pvheader['crval1']
    cdelt1 = pvheader['cdelt1']
    cdelt2 = pvheader['cdelt2']
    xvals = ((np.arange(naxis1)+1)-crpix1)*cdelt1 + crval1  # these will be in arcsec
    yvals = ((np.arange(naxis2)+1)-crpix2)*cdelt2 + crval2  # these in m/s
    yvals = yvals/1000. # m/s to km/s
    cdelt2 = cdelt2/1000. # likewise
    extent=[xvals[0]-cdelt1/2, xvals[-1]+cdelt1/2, yvals[0]-cdelt2/2, yvals[-1]+cdelt2/2] # outside edges of image
    chanwid_kms = np.abs(cdelt2)
    print(name,'Channel width kms',chanwid_kms)
    return pvim, xvals, yvals, cdelt1, chanwid_kms, extent

def calcratios(im1, im2, clipim, maskim, clipcase, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name):
    assert im1.shape == im2.shape, 'Images must be the same size.'
    assert im1.shape == clipim.shape, 'Clip image must be same size as others.'
    assert im1.shape == maskim.shape, 'Mask image must be same size as others.'
    ratios = []
    ratios_uncs = []
    midradii = [] # just for plotting
    ringwidth = [] # also just for plotting
    mask1 = 0.0*im1  # this one was actually just for display purposes so may not be relevant anymore
    nrings = len(ringboundaries) - 1
    pixperbeam2d = 1.1331 * fwhm**2 / (asecperpix**2)
    pixperbeam1d = 1.065 * fwhm / asecperpix
    mesh_x, mesh_y = np.meshgrid(xvals, yvals, sparse=False)
    
    if (clipcase == 1):
        # clipim is 12co 2"
        cliplevels= 7.9e-4 * np.array([2.5,3,6,12,20]) # is this scale factor the same as the rms noise in the pv?
    if (clipcase == 2):
        # clipim is 13co 2.6"
        cliplevels = 3.6e-4 * np.array([2.,3,6,12,20])  # these should be clip levels in the 13co image
    if (clipcase == 3):
        # clipim is 13co 2"
        cliplevels = 3.8e-4 * np.array([2.,3,6,12,20])  # these should be clip levels in the 13co image
    if (clipcase == 4):
        # clipcase 4 for cn32 2" 15kms
        cliplevels = 1.5e-4 * np.array([2.,3,6,12])
        bad = (mesh_y > 1100.)  # for the CN lines I need to exclude the other one by getting rid of the emission at the "wrong" velocities 
        maskim[bad] = 0.
    if (clipcase == 5):
        # clipcase 5 for the 2-1/1-0 line ratio at 1.1", clip on 1-0
        cliplevels = pvrms2 * np.array([1.,3.,5.,7.])
    
    ratio_b4stats = np.zeros((nrings,cliplevels.size)) # will hold values for all rings & all clip levels
    ratiounc_b4stats = np.zeros((nrings,cliplevels.size))  # likewise 
    sbprof1 = np.zeros((im1.shape[1],cliplevels.size)) # will hold a surface brightness profile which is the pv slice smashed along the velocity axis, multiple versions for clip levels
    sbprof2 = np.zeros((im2.shape[1],cliplevels.size)) # will hold a surface brightness profile which is the pv slice smashed along the velocity axis, multiple versions for clip levels
    # calculations
    # apply the clean mask to the clip image so we can ignore noisy pixels beyond the region of real emission
    clipim *= maskim
    # first will try making a surface brightness profile by integrating in velocity.
    # units?  in the pvslices themselves the values are Jy/beam and I think that is *averaged* over the 
    # slice - I've been doing 5 pix which is basically one beam FWHM.
    # then integrating over axis0, velocity, will give units (Jy/b)*channel
    # column density calculations will want (Jy/b)*(km/s), probably
    for j in range(cliplevels.size):
        temp_clip_mask = (clipim > cliplevels[j])
        sbprof1[:,j] = np.sum(im1*temp_clip_mask, axis=0)  # on these pv slices in python, axis0 is velocity and axis1 is spatial
        sbprof2[:,j] = np.sum(im2*temp_clip_mask, axis=0)  
    sbprof1unc = np.std(sbprof1, axis=1)
    sbprof1med = np.median(sbprof1, axis=1) # median of all the results for different clip levels.  maybe you like this, maybe you want the lowest clip level instead.
    sbprof2unc = np.std(sbprof2, axis=1)
    sbprof2med = np.median(sbprof2, axis=1)
    spaxwithdata = (sbprof1med > 0.)
    if (name=='cn32/cn12'):
        plt.figure()
        plt.errorbar(xvals[spaxwithdata], sbprof1med[spaxwithdata], yerr=sbprof1unc[spaxwithdata])
        ax = plt.gca()
        ax.set_yscale('log')
    # now do the ratios by summing in the regions of the pv slice
    for i in range(nrings):
        # loop over various clip values and use the multiple answers to define the "real" value and its unc
        # also check that clip values for the 12co comparison make sense
        for j in range(cliplevels.size):  # loop over clip levels
            useme = (np.abs(mesh_x) >= ringboundaries[i]) * (np.abs(mesh_x) < ringboundaries[i+1]) * (clipim > cliplevels[j]) # pixels in the ring and above some s/n ratio
            mask1[useme] = i + 1 + 0.1*j # this one was actually just for display purposes so may not be relevant anymore
            # uncertainties document recommends basically rms * sqrt(npix) * sqrt(pixperbeam) for a 2D sum
            # where the pixels are correlated in both spatial directions.  this pv is a little different
            # because pixels are correlated only in the spatial direction, not the frequency direction.
            # so a truly accurate error estimate should come probably from empirical estimates on some
            # actual pvslices with just noise in them.  I could probably do that but at the moment am
            # just gonna fake it by using the 1D pixperbeam factor rather than the 2D factor.
            sum1 = np.sum(im1[useme])
            sum2 = np.sum(im2[useme])
            npix = np.sum(useme)
            sum1_unc = pvrms1 * np.sqrt(npix) * np.sqrt(pixperbeam1d) # this method uses just rms in pv and # of pix in the pv image
            sum2_unc = pvrms2 * np.sqrt(npix) * np.sqrt(pixperbeam1d)
            ratio_b4stats[i,j] = sum1/sum2  # restfrequency corrections are below
            ratiounc_b4stats[i,j] = ratio_b4stats[i,j] * np.sqrt((sum1_unc/sum1)**2 + (sum2_unc/sum2)**2) # this one is based on npix
            if (name=='12co/hnco'):
                print('sum2 sum2_unc S/N %.4f %.4f %.1f'%(sum2,sum2_unc,(sum2/sum2_unc)))
        ratiomed = np.median(ratio_b4stats[i,:])
        # check which error is larger: the one based on npix or the one based on cliplevels?
        uncway1 = np.median(ratiounc_b4stats[i,:]) # this is the one based on npix
        uncway2 = np.std(ratio_b4stats[i,:]) # this is the one based on clip levels
        fullunc = np.sqrt(uncway1**2 + uncway2**2)
        print(ratio_b4stats[i,:])
        # careful when looking at these printout values because they have not been corrected to K units by the frequency ratios (that happens outside)
        print('Radii',ringboundaries[i],'to',ringboundaries[i+1],'ratio %.4f +- %.4f   npix %i'%(ratiomed,fullunc,npix))
        # sanity check: if there is signal in this region, write out results
        if ((sum1/sum1_unc >= 3.0) and (sum2/sum2_unc >= 3.0)):
            # note that if the clip levels are listed in ascending order like I usually do, these will be based on the npix at the highest clip level.
            midradii.append((ringboundaries[i]+ringboundaries[i+1])/2.)
            ringwidth.append((ringboundaries[i+1]-ringboundaries[i])/2.) # so it's actually the half-width cause that's what the plot routine wants
            ratios.append(ratiomed)
            ratios_uncs.append(fullunc)

    # sanity check plot
    plt.figure(name)
    imtoplot = im2 # most of the time this is the fainter one
    if (name=='12CO21_10'):
        imtoplot = im1
    if (name=='HNCO54_43'):
        imtoplot = im1  # already plotted the 4-3 line in another execution
    plt.imshow(imtoplot, origin='lower', extent=extent, interpolation='none', aspect='auto', norm=SymLogNorm(linthresh=pvrms2))
    plt.contour(clipim, origin='lower', extent=extent, levels=cliplevels, linewidths=0.5, colors='w')
    for k in range(1,nrings):
        plt.axvline(ringboundaries[k], color='w', linestyle=':')
        plt.axvline(-ringboundaries[k], color='w', linestyle=':')
    print()
    return midradii, ringwidth, np.array(ratios), np.array(ratios_uncs), sbprof1med, sbprof1unc, sbprof2med, sbprof2unc

def writemyfile(filename, midradii, ringwidth, ratios, ratios_uncs):
    outfile = open(filename, 'w')
    outfile.write("# {:%Y-%b-%d %H:%M:%S} by 4526pvslicefig.py\n".format(datetime.datetime.now()))
    outfile.write('# Midradius  Width   Ratio  Err \n')
    for i in range(len(midradii)):
        outfile.write('%5.2f %5.2f %7.3f %7.3f\n'%(midradii[i], ringwidth[i], ratios[i], ratios_uncs[i]))
    outfile.close()
                  
def writesbprof(filename, xvals, sb, sb_unc):
    outfile = open(filename, 'w')
    outfile.write("# {:%Y-%b-%d %H:%M:%S} by 4526pvslicefig.py\n".format(datetime.datetime.now()))
    outfile.write('# Position   Intensity  Intensity_unc  \n')
    outfile.write('#   (")      Jy/b*km/s  Jy/b*km/s  \n')
    for i in range(len(xvals)):
        outfile.write('%5.2f %7.3f %7.3f\n'%(xvals[i], sb[i], sb_unc[i]))
    outfile.close()
                  
                  
#%%
    
# figure of the 13co/hcn intensity ratio in the pv slices
pvim, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13cohcn.fits')
# make correction for the line ratio in K units rather than in Jy units; see notebook
pvim *= (restfreq_hcn/restfreq_13co)**2
plt.figure('pv')
plt.imshow(np.log10(pvim), origin='low', vmin=0.0, vmax=0.8, extent=extent,\
           interpolation='nearest', aspect='auto')
plt.tick_params(direction='in', top=True, right=True)
plt.xlabel('Arcsec')
plt.ylabel('Velocity (km/s)')
cb = plt.colorbar()
cb.set_label('log (13CO/HCN)')
plt.savefig('pvratio.pdf')
# updated for high res  13co/c18o maybe or 12co/c18o?  are these obvious? 12co/13co is not variable and c18o kinda low S/N
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_13co_15kms_morechans.fits')
im2, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_c18o_15kms_morechans.fits')
ratioim = np.log10(im1/im2)
ratioim[im1 <= 3.*3.8e-4] = np.nan
plt.figure('pv2')
plt.imshow(ratioim, origin='low', vmin=0.3, vmax=1.0, extent=extent, aspect='auto')
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_sm.fits')
im2, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hcn_sm.fits')
ratioim = np.log10((im1/im2)*(restfreq_hcn/restfreq_12co)**2)
ratioim[im2 <= 2.9e-4] = np.nan # 1sigma
ratioim[im1 <= 2*1.1e-3] = np.nan # 2sigma
plt.figure('pv3')
plt.imshow(ratioim, origin='low', vmin=0.45, vmax=1.3, extent=extent, aspect='auto', interpolation='nearest')
plt.tick_params(direction='in', top=True, right=True)
cb = plt.colorbar()
cb.set_label('log (12CO/HCN)')
plt.xlabel('Offset (")')
plt.ylabel('Velocity (km s$^{-1}$)')
plt.savefig('pvratio2.pdf')



#%%

# radial variation in the ratios by selecting pixels in the PV slices

# will try using the 13co image for defining pv regions.  should only need to do this once.
im13co, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13co_sm.fits') # this one is 2.6" resolution I'm pretty sure
# clean mask is also used in the pixel selection process? (I think)
mask13co, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13comask.fits')
# new high res 12co data and corresponding 2" 13co data
im12co, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_2asec_15kms.fits')
mask12co, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12comask_2asec_15kms.fits')
im13conew, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_13co_15kms_morechans.fits')
mask13conew, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_13comask_15kms_morechans.fits')
maskcn32, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_cn32mask_2asec_15kms.fits')

# 13co/hcn
name='13co/hcn'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hcn_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
# need to get the rms values for all the different lines.  I'm not sure I want to derive that
# every time, might just hard code it.
# remember also if you are just reading off the casa image, that's the rms in the *average* over the slit
pvrms1 = 3.6e-4 # Jy/beam; 13co. get this directly off the pv image in casa. so 3sigma is 0.0011
pvrms2 = 2.9e-4 # Jy/beam; HCN
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc  = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hcn/restfreq_13co)**2
ratios_uncs *= (restfreq_hcn/restfreq_13co)**2
# write results to a file
writemyfile('pvratios_13CO_HCN.txt', midradii, ringwidth, ratios, ratios_uncs)




# now will try radial variation in 13co/cs
name='13co/cs'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cs_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 3.6e-4 # Jy/beam; 13co. so 3sigma is 0.0011
pvrms2 = 2.8e-4 # Jy/beam; cs
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb13co, sb13co_unc, sbcs, sbcs_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cs/restfreq_13co)**2
ratios_uncs *= (restfreq_cs/restfreq_13co)**2
# write results to a file
writemyfile('pvratios_13CO_CS.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbcs *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbcs_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_CS.txt', xvals, sbcs, sbcs_unc)
# 13co surface brightness profile is done later when I'm dealing with higher resolution data

# now will try radial variation in 13co/ch3oh
name='13co/ch3oh'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_13co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_ch3oh_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 3.6e-4 # Jy/beam; 13co. so 3sigma is 0.0011
pvrms2 = 2.9e-4 # Jy/beam; ch3oh
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb13co, sb13co_unc, sbch3oh, sbch3oh_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_ch3oh/restfreq_13co)**2
ratios_uncs *= (restfreq_ch3oh/restfreq_13co)**2
# write results to a file
writemyfile('pvratios_13CO_CH3OH.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbch3oh *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbch3oh_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_CH3OH.txt', xvals, sbch3oh, sbch3oh_unc)
# 13co surface brightness profile is done later when I'm dealing with higher resolution data

# how about HCN/HCO+?
name='hcn/hco+'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hcn_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hco+_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; hcn
pvrms2 = 2.8e-4 # Jy/beam; hco+
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sbhcn, sbhcn_unc, sbhcop, sbhcop_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hcop/restfreq_hcn)**2
ratios_uncs *= (restfreq_hcop/restfreq_hcn)**2
# write results to a file
writemyfile('pvratios_HCN_HCO+.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbhcn *= chanwid_kms  #  profile now in Jy/b*km/s.
sbhcn_unc *= chanwid_kms # likewise
sbhcop *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbhcop_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_HCN.txt', xvals, sbhcn, sbhcn_unc)
writesbprof('SBprof_HCO+.txt', xvals, sbhcop, sbhcop_unc)

# how about HCN/HNC?
name='hcn/hnc'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hcn_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hnc_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; hcn
pvrms2 = 2.6e-4 # Jy/beam; hnc
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sbhcn, sbhcn_unc, sbhnc, sbhnc_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hnc/restfreq_hcn)**2
ratios_uncs *= (restfreq_hnc/restfreq_hcn)**2
# write results to a file
writemyfile('pvratios_HCN_HNC.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbhcn *= chanwid_kms  #  profile now in Jy/b*km/s.
sbhcn_unc *= chanwid_kms # likewise
sbhnc *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbhnc_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_HNC.txt', xvals, sbhnc, sbhnc_unc)

# how about HCN/HNCO?
name='hcn/hnco'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hcn_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hnco.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; hcn
pvrms2 = 2.6e-4 # Jy/beam; hnc
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sbhcn, sbhcn_unc, sbhnco, sbhnco_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hnco/restfreq_hcn)**2
ratios_uncs *= (restfreq_hnco/restfreq_hcn)**2
# write results to a file
writemyfile('pvratios_HCN_HNCO.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbhcn *= chanwid_kms  #  profile now in Jy/b*km/s.
sbhcn_unc *= chanwid_kms # likewise
sbhnco *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbhnco_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_HNCO.txt', xvals, sbhnco, sbhnco_unc)

# how about HCN/CS?
name='hcn/cs'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hcn_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cs_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; hcn
pvrms2 = 2.8e-4 # Jy/beam; cs
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cs/restfreq_hcn)**2
ratios_uncs *= (restfreq_cs/restfreq_hcn)**2
# write results to a file
writemyfile('pvratios_HCN_CS.txt', midradii, ringwidth, ratios, ratios_uncs)

# how about HCO+/CS?
name='hco+/cs'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hco+_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cs_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; hcn
pvrms2 = 2.8e-4 # Jy/beam; cs
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cs/restfreq_hcop)**2
ratios_uncs *= (restfreq_cs/restfreq_hcop)**2
# write results to a file
writemyfile('pvratios_HCO+_CS.txt', midradii, ringwidth, ratios, ratios_uncs)

# how about ch3oh/hnco?
name='ch3oh/hnco'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_ch3oh_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hnco.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; ch3oh
pvrms2 = 2.9e-4 # Jy/beam; hnco
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hnco/restfreq_ch3oh)**2
ratios_uncs *= (restfreq_hnco/restfreq_ch3oh)**2
# write results to a file
writemyfile('pvratios_CH3OH_HNCO.txt', midradii, ringwidth, ratios, ratios_uncs)

# how about ch3oh/hcn?
name='ch3oh/hcn'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_ch3oh_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hcn_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.9e-4 # Jy/beam; ch3oh
pvrms2 = 2.9e-4 # Jy/beam; hcn
fwhm = 2.63 # arcsec; they were all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hcn/restfreq_ch3oh)**2
ratios_uncs *= (restfreq_hcn/restfreq_ch3oh)**2
# write results to a file
writemyfile('pvratios_CH3OH_HCN.txt', midradii, ringwidth, ratios, ratios_uncs)


# 12co/13co
name='12co/13co'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_2asec_15kms.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_13co_15kms_morechans.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 7.9e-4 # Jy/beam; 12co
pvrms2 = 3.8e-4 # Jy/beam; 13co @ 2"
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the 12co image
midradii, ringwidth, ratios, ratios_uncs, sb12co, sb12co_unc, sb13co, sb13co_unc = calcratios(im1, im2, im12co, mask12co, 1, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_13co/restfreq_12co)**2
ratios_uncs *= (restfreq_13co/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_13CO.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sb12co *= chanwid_kms  #  profile now in Jy/b*km/s.
sb12co_unc *= chanwid_kms # likewise
writesbprof('SBprof_12CO.txt', xvals, sb12co, sb12co_unc)


# 12co/c18o
name='12co/c18o'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_2asec_15kms.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_c18o_15kms_morechans.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 7.9e-4 # Jy/beam; 12co
pvrms2 = 3.1e-4 # Jy/beam; c18o @ 2"
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the 12co image
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im12co, mask12co, 1, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_c18o/restfreq_12co)**2
ratios_uncs *= (restfreq_c18o/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_C18O.txt', midradii, ringwidth, ratios, ratios_uncs)


# 13co/c18o at 2" res
name='13co/c18o'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_13co_15kms_morechans.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_c18o_15kms_morechans.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.0] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 3.8e-4 # Jy/beam; 13co
pvrms2 = 3.1e-4 # Jy/beam; c18o @ 2"
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the 13co image
midradii, ringwidth, ratios, ratios_uncs, sb13co, sb13co_unc, sbc18o, sbc18o_unc = calcratios(im1, im2, im13conew, mask13conew, 3, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_c18o/restfreq_13co)**2
ratios_uncs *= (restfreq_c18o/restfreq_13co)**2
# write results to a file
writemyfile('pvratios_13CO_C18O.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sb13co *= chanwid_kms  #  profile now in Jy/b*km/s.
sb13co_unc *= chanwid_kms # likewise
writesbprof('SBprof_13CO.txt', xvals, sb13co, sb13co_unc)
sbc18o *= chanwid_kms_2  #  profile now in Jy/b*km/s.
sbc18o_unc *= chanwid_kms_2 # likewise
writesbprof('SBprof_C18O.txt', xvals, sbc18o, sbc18o_unc)

#%%
# cn32/cn12 at high res
name='cn32/cn12'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_cn32_2asec_15kms.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cn12_2asec_15kms.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.] # 
pvrms1 = 1.7e-4 # Jy/beam; cn32 2" 45 kms channels, not entirely accurate for interpolated channels
pvrms2 = 1.7e-4 # Jy/beam; 
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask - note here I am using the 12co image for a mask, as neither of the CNs is very high S/N but both have been regridded to match 12co.  this should clip out the other line from the pv.
midradii, ringwidth, ratios, ratios_uncs, sbcn32, sbcn32_unc, sbcn12, sbcn12_unc = calcratios(im1, im2, im12co, mask12co, 1, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cn12/restfreq_cn32)**2
ratios_uncs *= (restfreq_cn12/restfreq_cn32)**2
# write results to a file
writemyfile('pvratios_CN32_CN12.txt', midradii, ringwidth, ratios, ratios_uncs)
# prep the surface brightness profiles for column density calculations.
sbcn32 *= chanwid_kms  #  profile now in Jy/b*km/s.
sbcn32_unc *= chanwid_kms # likewise
writesbprof('SBprof_CN32.txt', xvals, sbcn32, sbcn32_unc)
#%%

# 12co/cn32.  cn32 is suppoased to be regridded to match 12co.
name='12co/cn32'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_2asec_15kms.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cn32_2asec_15kms.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.] # 
pvrms1 = 7.9e-4 # Jy/beam;
pvrms2 = 1.7e-4 # Jy/beam; this was measured in 45kms channels; some subtleties now about non-independent channels but will ignore temporarily
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im12co, mask12co, 1, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cn32/restfreq_12co)**2
ratios_uncs *= (restfreq_cn32/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_CN32.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co/cn32 version 2, clipping on the cn32 image itself because of fatter line (HF components)
name='12co/cn32v2'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_2asec_15kms.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cn32_2asec_15kms.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 15.] # 
pvrms1 = 7.9e-4 # Jy/beam;
pvrms2 = 1.7e-4 # Jy/beam; this was measured in 45kms channels; some subtleties now about non-independent channels but will ignore temporarily
fwhm = 2.0 # arcsec; these two are matched resolution 2.0"
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask on CN32
copyofim2 = 1.0*im2
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, copyofim2, maskcn32, 4, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cn32/restfreq_12co)**2
ratios_uncs *= (restfreq_cn32/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_CN32v2.txt', midradii, ringwidth, ratios, ratios_uncs)


# 12co/hcn.  adding 3/3/2021 after smoothing 12co to 2.63" like smoothed hcn.
name='12co/hcn'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hcn_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.1e-3 # Jy/beam; smoothed 12co
pvrms2 = 2.9e-4 # Jy/beam; HCN
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hcn/restfreq_12co)**2
ratios_uncs *= (restfreq_hcn/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_HCN.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co/cs.  adding 3/3/2021 after smoothing 12co to 2.63" like smoothed hcn.
name='12co/cs'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cs_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.1e-3 # Jy/beam; smoothed 12co
pvrms2 = 2.8e-4 # Jy/beam; cs
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cs/restfreq_12co)**2
ratios_uncs *= (restfreq_cs/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_CS.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co/ch3oh.  adding 3/3/2021 after smoothing 12co to 2.63" like smoothed hcn.
name='12co/ch3oh'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_ch3oh_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.1e-3 # Jy/beam; smoothed 12co
pvrms2 = 2.9e-4 # Jy/beam; ch3oh
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_ch3oh/restfreq_12co)**2
ratios_uncs *= (restfreq_ch3oh/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_CH3OH.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co/hnco
name='12co/hnco'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hnco.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.1e-3 # Jy/beam; smoothed 12co
pvrms2 = 2.6e-4 # Jy/beam; hnco
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hnco/restfreq_12co)**2
ratios_uncs *= (restfreq_hnco/restfreq_12co)**2
# write results to a file
writemyfile('pvratios_12CO_HNCO.txt', midradii, ringwidth, ratios, ratios_uncs)

# cn/hcn 
name='CN/HCN'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_cn32_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_hcn_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 3.0e-4 # Jy/beam; smoothed cn32
pvrms2 = 2.9e-4 # Jy/beam; hcn
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hcn/restfreq_cn32)**2
ratios_uncs *= (restfreq_hcn/restfreq_cn32)**2
# write results to a file
writemyfile('pvratios_CN32_HCN.txt', midradii, ringwidth, ratios, ratios_uncs)

# cn/cs 
name='CN/CS'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_cn32_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfile('pv_cs_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 3.0e-4 # Jy/beam; smoothed cn32
pvrms2 = 2.8e-4 # Jy/beam; cs
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cs/restfreq_cn32)**2
ratios_uncs *= (restfreq_cs/restfreq_cn32)**2
# write results to a file
writemyfile('pvratios_CN32_CS.txt', midradii, ringwidth, ratios, ratios_uncs)

# cs/cn
name='CS/CN'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_cs_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_cn32_sm.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.8e-4 # Jy/beam; cs
pvrms2 = 3.0e-4 # Jy/beam; smoothed cn32
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_cn32/restfreq_cs)**2
ratios_uncs *= (restfreq_cn32/restfreq_cs)**2
# write results to a file
writemyfile('pvratios_CS_CN32.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co 2-1/1-0
name='12CO21_10'
maskfor2110, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co_maskfor21.fits')
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co21_sm.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_12co_for21.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 0.55, 1.65, 2.75, 3.85, 4.95, 6.05, 7.15, 8.25, 9.35, 10.45, 11.55, 15] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.04e-2 # Jy/beam; 2-1
pvrms2 = 1.49e-3 # Jy/beam; 1-0
fwhm = 1.12 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.3 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# working here. clip on 12co, obviously, but as this is regridded from all the other versions you will need yet another mask/clip set!
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im2, maskfor2110, 5, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_12co/restfreq_12co21)**2
ratios_uncs *= (restfreq_12co/restfreq_12co21)**2
# write results to a file
writemyfile('pvratios_12CO21_10.txt', midradii, ringwidth, ratios, ratios_uncs)

# 12co 2-1/1-0 redo with continuum subtraction
name='12CO21_10'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfilevel('pv_12co21_sm_contsub.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_12co_for21.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 0.55, 1.65, 2.75, 3.85, 4.95, 6.05, 7.15, 8.25, 9.35, 10.45, 11.55, 15] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 1.04e-2 # Jy/beam; 2-1
pvrms2 = 1.49e-3 # Jy/beam; 1-0
fwhm = 1.12 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.3 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# working here. clip on 12co, obviously, but as this is regridded from all the other versions you will need yet another mask/clip set!
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im2, maskfor2110, 5, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_12co/restfreq_12co21)**2
ratios_uncs *= (restfreq_12co/restfreq_12co21)**2
# write results to a file
writemyfile('pvratios_12CO21_10_contsub.txt', midradii, ringwidth, ratios, ratios_uncs)

# HNCO54/43
name='HNCO54_43'
im1, xvals, yvals, cdelt1, chanwid_kms, extent = readfile('pv_hnco54_26asec.fits')
im2, xvals2, yvals2, cdelt1_2, chanwid_kms_2, extent2 = readfilevel('pv_hnco.fits')
slitwidth = 5 # number of pixels averaged across the slit (see casa reduction steps)
ringboundaries = [0.0, 1.3, 3.9, 6.6, 9.2, 11.8, 16] # in arcsec; outer edges of regions to use for measurements
pvrms1 = 2.8e-4 # Jy/beam; HNCO54 
pvrms2 = 2.9e-4 # Jy/beam; hnco
fwhm = 2.63 # arcsec; all smoothed to match the lowest resolution line
asecperpix = 0.4 # arcsec; linear size; this was also specified at imaging 
# clip & mask based on the brighter image
# I havent yet made a 12co mask image at 2.63" resolution.  could cheez out and use 13co versions prepared earlier for this situation at 2.63".
midradii, ringwidth, ratios, ratios_uncs, sb1, sb1_unc, sb2, sb2_unc = calcratios(im1, im2, im13co, mask13co, 2, xvals, yvals, slitwidth, ringboundaries, pvrms1, pvrms2, fwhm, asecperpix, name)
# make correction for the line ratio in K units rather than in Jy units; see notebook
ratios *= (restfreq_hnco/restfreq_hnco54)**2
ratios_uncs *= (restfreq_hnco/restfreq_hnco54)**2
# write results to a file
writemyfile('pvratios_HNCO54_43.txt', midradii, ringwidth, ratios, ratios_uncs)


