#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:43:55 2019

@author: lyoung

PURPOSE: Here I'm experimenting with calculating line ratios from the mom0s and plotting individual
   pixels' ratios vs radius.  Will use the deprojected radius since I know the inclination.
   Also reads the line ratio files produced by the other scripts that use different methods (e.g. annular spectra).
   Produces a large number of figures.
    
  Initial version modified from 4526lineratios4.py  because it already has a lot of the geometry code.
  Plus lifting some from 4526plotradialratios.py to compare with the data from the other methods.
  

  
TO DO:


"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.stats import mad_std
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from matplotlib.ticker import LogLocator
from matplotlib import colors

def tedious_repetitive(ylabel):
    ax_kpc = ax_asec.twin(aux_trans)
    ax_kpc.set_viewlim_mode("transform")
    ax_asec.axis['bottom'].set_label('Radius (")')
    ax_asec.axis['left'].set_label(ylabel)
    ax_asec.axis['left'].label.set_visible(True)
    ax_kpc.axis["top"].set_label('Radius (kpc)')
    ax_kpc.axis["top"].label.set_visible(True)
    ax_kpc.axes.tick_params(which='both', direction='in')
    ax_asec.axis["right"].major_ticklabels.set_visible(False)
    ax_kpc.axis["right"].major_ticklabels.set_visible(False)
    ax_asec.set_yscale('log')
    ax_asec.get_yaxis().set_ticks_position('both')
    ax_asec.axes.tick_params(which='both', direction='in')
    ax_asec.get_yaxis().set_tick_params(which='major', length=8)
    y_major = LogLocator(base = 10.0, numticks = 5)
    ax_asec.get_yaxis().set_major_locator(y_major)
    ax_kpc.get_yaxis().set_major_locator(y_major)
    y_minor = LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax_asec.get_yaxis().set_minor_locator(y_minor)
    ax_kpc.get_yaxis().set_minor_locator(y_minor)
    ax_asec.legend(loc='best', frameon=False, handlelength=0)


def plotme(filename, label, color, dooffset):
    if 'annuli' in filename:
        annuli = np.genfromtxt(filename, usecols=[0,1,2,3], names=['inner','outer','ratio','err'], \
                       dtype=None)
        offsets = np.zeros(annuli.size)
        if (dooffset):
            offsets += 0.5 * np.random.random_sample(annuli.size)
        xvals = (annuli['inner']+annuli['outer'])/2. + offsets
        ax_asec.errorbar(xvals, annuli['ratio'], yerr=annuli['err'], xerr=(annuli['outer']-annuli['inner'])/2.,\
             fmt='o', color=color, markerfacecolor='None', label=label)
    else:
        pvs = np.genfromtxt(filename, usecols=[0,1,2,3], names=['mid','width','ratio','err'], dtype=None)
        if 'slant' in filename:
            fmt = '^'
        else:
            fmt = 'o'
        ax_asec.errorbar(pvs['mid'], pvs['ratio'], yerr=pvs['err'], xerr=pvs['width'], fmt=fmt, color=color, label=label)
        
        
def calcmomratios(line1, line1err, line2, line2err, freq1, freq2, name1, name2, cm):
    print('')
    print(name1+'/'+name2)
    
    line1file = fits.open(myDir+line1)  # these should have units Jy/b*km/s unless it is continuum when they will be Jy/beam
    line1err = fits.open(myDir+line1err)[0].data
    im1 = line1file[0].data 
    w = wcs.WCS(line1file[0].header)
    bmaj1 = float(line1file[0].header['BMAJ']) # degrees
    bmin1 = float(line1file[0].header['BMIN'])
    beamarea_sr = 1.1331 * bmaj1 * bmin1 * (np.pi/180.)**2 
    bmaj1 *= 3600. # now in arcsec
    bmin1 *= 3600. 
    asecperpix1 = float(line1file[0].header['CDELT1'])
    line1file.info()
    
    line2file = fits.open(myDir+line2)  # these should have units Jy/b*km/s
    im2 = line2file[0].data  
    bmaj2 = float(line2file[0].header['BMAJ'])*3600.
    bmin2 = float(line2file[0].header['BMIN'])*3600.
    asecperpix2 = float(line2file[0].header['CDELT1'])
    line2err = fits.open(myDir+line2err)[0].data
    
    #  the dust image gets slightly different treatment since the uncertainty is mostly constant everywhere and I don't have a "formal" error image
    if (name1=='dust'): 
        line1err = 0.0*im1 + 1.25e-5  # Jy/beam
        # also in this case I want to convert the line image (CO) to K km/s
        lam2 = c/(freq2*1e9) # now in meters for im2, the line image
        im2 *= 13.6 * (lam2*1e3)**2 / (bmaj2*bmin2) # im2 is now in K km/s
        line2err *= 13.6 * (lam2*1e3)**2 / (bmaj2*bmin2) # line2err is now in K km/s
        # and convert the continuum image to Msun(of H2)/pc2
        nu_cont = 99.33e9 # Hz.  this is for cont_bigmfs_2asec_2t
        lam_cont = c/nu_cont # in meters
        BnuT = 2.* h * (nu_cont)**3 / (c**2 * (np.exp(h*nu_cont/(kb*Tdust))-1.)) # SI units, mks
        opacity = opacity_coef * (1.e-6*scalewave/lam_cont)**Beta  
        factor1 = 1.e-26 / beamarea_sr # Jy/b to W/m2.Hz.sr
        factor2 = gasdust / (BnuT * opacity) # W/m2.Hz.sr to kg/m2. 
        factor3 = (3.09e16)**2 / 1.99e30 # kg/m2 to  Msun (of H2, not He) / pc2
        im1 *= factor1 * factor2 * factor3 # now Msun/pc2
        line1err *= factor1 * factor2 * factor3
        print('conversion factor for dust image to Msun/pc2 is %7.3e'%(factor1*factor2*factor3))
        factor4 = 6.24e19 # Msun/pc2 to molecules/cm2 (i.e. 6e19 mol/cm2 per Msun/pc2)
        print('conversion factor for dust image to N(H2) in cm2 is %7.3e'%(factor1*factor2*factor3*factor4))
        factor5 = 13.6 * (lam_cont*1e3)**2 / (bmaj1*bmin1) # Jy/b to K
        print('conversion factor for dust image to K is %7.3e'%(factor5))
        
    line2file.info()
    assert im1.shape == im2.shape, 'Are these the same region?'
    assert asecperpix1 == asecperpix2, 'Do these pixels line up?' 
    print('beam1 (") %.3f'%bmaj1)
    print('beam2 (") %.3f'%bmaj2)
    #
    xvals_pix = np.arange(im1.shape[1])
    yvals_pix = np.arange(im1.shape[0])  
    x_mesh_pix, y_mesh_pix = np.meshgrid(xvals_pix, yvals_pix, sparse=False)
    x_mesh_world, y_mesh_world = w.wcs_pix2world(x_mesh_pix, y_mesh_pix, 0) # returned values are in degrees
    x_mesh_rel = x_mesh_world - centerra  # now in degrees relative to chosen center
    y_mesh_rel = y_mesh_world - centerdec
    # usually I want to use the deprojected radial distance along the disk
    costh = np.cos(theta)
    sinth = np.sin(theta)
    x2_mesh_rot = (costh**2 * x_mesh_rel**2) + (sinth**2 * y_mesh_rel**2) + (2 * x_mesh_rel * y_mesh_rel * costh * sinth)
    y2_mesh_rot = (sinth**2 * x_mesh_rel**2) + (costh**2 * y_mesh_rel**2) - (2 * x_mesh_rel * y_mesh_rel * costh * sinth)
    dist2 = x2_mesh_rot + (y2_mesh_rot/np.cos(inc*np.pi/180))  # (distance)^2, in deg; usage: < sma**2 is inside the ellipse
    dist_asec = np.sqrt(dist2) * 3600.
    # for the work with the continuum images I also need a projected distance (in the map plane)
    # so I can exclude pixels contaminated by the synchrotron point source
    projecteddist = np.sqrt(x_mesh_rel**2 + y_mesh_rel**2) * 3600. # now in arcsec
    # ratio
    useme = (((im2/line2err) >= 1.0) * ((im1/line1err) >= 1.0))  # multiply is logical and
    if (name1=='dust'): 
        # redo the above, as we need a more stringent S/N cutoff on the dust continuum image (there's no other masking on it)
        useme = (((im2/line2err) >= 1.0) * ((im1/line1err) >= 3.0))  # 3sigma on the dust
        # also exclude pixels with synchrotron contamination
        useme *= (projecteddist >= 1.3*bmaj1)  # as bmaj is fwhm, I'm effectively taking radius > 2*HWHM which ought to be fairly conservative
    ratio = (im1/im2) * (freq2/freq1)**2 # latter part converts this to line ratios in K units.  Note that for the alpha calculation we pass freq1 = freq2 so this has no effect, which is what we want, as the numbers are fixed elsewhere
    ratioerr = ratio * np.sqrt((line2err/im2)**2 + (line1err/im1)**2)
    ratiosn = ratio/ratioerr
    minsn = np.min(ratiosn[useme])
    maxsn = np.max(ratiosn[useme])
    alphascale = (ratiosn-minsn)/(maxsn-minsn)
    #
    dist_asec_trim = dist_asec[useme]  # klutziness for enabling the variable alphas
    ratio_trim = ratio[useme]
    ratioerr_trim = ratioerr[useme]
    cmap = colors.LinearSegmentedColormap.from_list('incr_alpha', [(0, (*colors.to_rgb(cm),0)), (1, cm)])
    plt.scatter(dist_asec_trim, ratio_trim, c=alphascale[useme], cmap=cmap, ec=None, s=30, label=name1+'/'+name2)    
    #
    # some stats
    useme2 = (((im2/line2err) >= 3.0) * ((im1/line1err) >= 3.0))  # just the high S/N set for statistics
    if (name1=='dust'):
        # remove the synchrotron contamination from the stats
        useme2 *= (projecteddist >= 1.3*bmaj1)
        # now also bin the ratios and plot errorbars for the bins
        bincenters = [3.5,5.5,7.5,9.5]
        binwidths = 1.0 # half-width in asec
        binmedians = []
        binerrs = []
        for i in range(len(bincenters)):
            binme = (dist_asec_trim >= (bincenters[i]-binwidths)) * (dist_asec_trim < (bincenters[i]+binwidths))
            binmedians.append(np.median(ratio_trim[binme]))
            binerrs.append(mad_std(ratio_trim[binme]))
        plt.errorbar(bincenters, binmedians, xerr=binwidths, yerr=binerrs, fmt='k.')
        
    blah = mad_std(ratio[useme2])
    blah2 = np.std(ratio[useme2])
    print('Median and std of this ratio:',np.median(ratio[useme2]),'+-',blah2,blah)
    # sanity check on the distance calculations
    plt.figure()
    plt.imshow(im1, origin='lower')
    plt.contour(dist_asec, origin='lower', levels=[5,10,15,20,25,30])
    
    return ratio_trim, ratioerr_trim
    
# ----------- setup -----------------------
    

# definitions
restfreq_hcn = 88.631601
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
h = 6.62e-34 # J s
c = 3.0e8 # m/s
kb = 1.38e-23 # J/K
gasdust = 100 # this is m(H2)/m(dust) -- not including He
doAuld = False # which MBB dust model?
if (doAuld):
    # Auld & co used opacity 0.192 m2/kg @ 350 micron, with a slope of 2
    Tdust = 23.8 # fitted temp in K
    Beta = 2.0
    opacity_coef = 0.192 # m2/kg
    scalewave = 350 # wavelength (in microns) at which the opacity is measured
    dist = 17 # in Mpc assumed as part of the fit
    dustmass = 1.0e7 # in solar masses
else:
    # Dustpedia values are slightly different
    Tdust = 25.1
    Beta = 1.790
    opacity_coef = 0.640
    scalewave = 250
    dist = 15.35 # seems low, but whatever
    dustmass = 4.38e6

dosbprof = True # overlay so we can see structure of molecular disk with ring etc
dovline = True # marking location of the peak SB on the ring
vloc = 5.2 # arcsec, for the line mentioned above


# other generic setup
myDir = 'Dropbox/projects/alma4526/'

# adopted disk parameters for ngc4526
inc = 79 # disk inclination in degrees
theta = -22*np.pi/180  # rotation angle in radians
centerra = 15.0 * (12.0 + 34.0/60 + 02.997/3600) # 12h34m02.997s converted to degrees
centerdec = 7.0 + 41.0/60 + 57.87/3600 # 7d41m57.87s converted to decimal degrees
ringboundaries = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0])  # semimajor axes, in arcsec
ringboundaries_deg = ringboundaries/3600.
nrings = len(ringboundaries) - 1

# linear scale parameters
asecperkpc = 1000./(4.85*16.4) # 16.4 Mpc for 4526
aux_trans = mtransforms.Affine2D().scale(asecperkpc, 1.)


#  ----------- start work here -------------

doalltheintermediatestuff = True
if (doalltheintermediatestuff):

    # CO isotope line ratios
    fig5 = plt.figure()
    ax_asec = SubplotHost(fig5, 1, 1, 1, aspect='auto')
    fig5.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('12co_2asec.integrated.fits','12co_2asec.integrated.m0err.fits','c18o_masked.m0.fits','c18o_masked.m0err.fits',restfreq_12co,restfreq_c18o,'$^{12}$CO','C$^{18}$O', 'C2')
    ratio, ratioerr = calcmomratios('13co_masked.m0.fits','13co_masked.m0err.fits','c18o_masked.m0.fits','c18o_masked.m0err.fits',restfreq_13co,restfreq_c18o,'$^{13}$CO','C$^{18}$O', 'C1')
    ratio, ratioerr = calcmomratios('12co_2asec.integrated.fits','12co_2asec.integrated.m0err.fits','13co_masked.m0.fits','13co_masked.m0err.fits',restfreq_12co,restfreq_13co,'$^{12}$CO','$^{13}$CO', 'C0') #the last entry is a colormap name
    plotme(myDir+'pvratios_13CO_C18O.txt', '', 'black', False)
    plotme(myDir+'annuli_13CO_C18O.txt', '', 'black', False)
    plotme(myDir+'pvratios_12CO_C18O.txt', '', 'black', False)
    plotme(myDir+'annuli_12CO_C18O.txt', '', 'black', False)
    plotme(myDir+'pvratios_12CO_13CO.txt', '', 'black', False)
    plotme(myDir+'annuli_12CO_13CO.txt', '', 'black', False)
    plt.ylim(2, 50.)
    plt.xlim(0, 15.)
    tedious_repetitive('Ratio')
    textcolors=['C2','C1','C0'] 
    leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
    for i in enumerate(leg.get_texts()):
         i[1].set_color(textcolors[i[0]])
    if (dosbprof):
        sbprof = np.genfromtxt('hires12comom0slice.txt', usecols=[0,1], names=['off','sb'])
        scalefactor = 45./np.nanmax(sbprof['sb'])
        plt.plot(sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey')
        plt.plot(-sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey') # arbitrary scale factor to get the lines to show up on the plot
    if (dovline):
        plt.axvline(vloc, color='k', linestyle=':')
    fig5.savefig('scatterratios.pdf')
    
    
    # CN and HCN line ratios
    fig3 = plt.figure()
    ax_asec = SubplotHost(fig3, 1, 1, 1, aspect='auto')
    fig3.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('cn32_26asec.m0v1.fits', 'cn32_2asec.m0err.fits', 'hcn_26asec.m0.fits', 'hcn_masked.m0err.fits', restfreq_cn32, restfreq_hcn, 'CN32','HCN','C2')
    plotme(myDir+'pvratios_CN32_HCN.txt', '', 'black', False)
    plotme(myDir+'annuli_CN32_HCN.txt', '', 'black', False)
    ratio, ratioerr = calcmomratios('cn32_na.mom0.fits','cn32_na.m0err.fits','cn12_na.mom0.fits','cn12_na.m0err.fits',restfreq_cn32,restfreq_cn12,'CN32','CN12', 'C4')
    ratio, ratioerr = calcmomratios('12co_2asec.integrated.fits','12co_2asec.integrated.m0err.fits','cn32_2asec.mom0.fits','cn32_2asec.m0err.fits',restfreq_12co,restfreq_cn32,'12CO','CN32', 'C9')
    plotme(myDir+'pvratios_CN32_CN12.txt', '', 'black', False)
    plotme(myDir+'annuli_CN32_CN12.txt', '', 'black', False)
    plotme(myDir+'annuli_12CO_CN32.txt', '', 'black', False)
    plotme(myDir+'pvratios_12CO_CN32.txt', '', 'black', False)
    plt.xlim(0,15)
    plt.ylim(0.3,70)
    plt.axhline(2.0, linestyle='--', color='mediumorchid')
    tedious_repetitive('Ratio')
    textcolors=['forestgreen','mediumorchid','xkcd:turquoise']
    leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
    for i in enumerate(leg.get_texts()):
         i[1].set_color(textcolors[i[0]])
    if (dosbprof):
        scalefactor = 65./np.nanmax(sbprof['sb'])
        plt.plot(sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey')
        plt.plot(-sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey') # arbitrary scale factor to get the lines to show up on the plot
    if (dovline):
        plt.axvline(vloc, color='k', linestyle=':')
    fig3.savefig('scatterratios_cn.pdf')
    
    
    
    fig5 = plt.figure()
    ax_asec = SubplotHost(fig5, 1, 1, 1, aspect='auto')
    fig5.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('hcn_masked.m0.fits','hcn_masked.m0err.fits','hnc_masked.m0.fits','hnc_masked.m0err.fits',restfreq_hcn,restfreq_hnc,'HCN','HNC', 'C0')
    plotme(myDir+'annuli_HCN_HNC.txt', '', 'k', False)
    plotme(myDir+'pvratios_HCN_HNC.txt', '', 'k', False)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    
    fig6 = plt.figure()
    ax_asec = SubplotHost(fig6, 1, 1, 1, aspect='auto')
    fig6.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('hcn_26asec.m0.fits','hcn_masked.m0err.fits','cs_26asec.m0.fits','cs_masked.m0err.fits',restfreq_hcn,restfreq_cs,'HCN','CS', 'C0')
    ratio, ratioerr = calcmomratios('cs_26asec.m0.fits','cs_masked.m0err.fits','hcn_26asec.m0.fits','hcn_masked.m0err.fits',restfreq_cs,restfreq_hcn,'CS','HCN', 'C1')
    plotme(myDir+'annuli_HCN_CS.txt', '', 'k', False)
    plotme(myDir+'pvratios_HCN_CS.txt', '', 'k', False)
    plt.xlim(0,15)
    plt.ylim(0.05,20)
    tedious_repetitive('Ratio')
    
    fig7 = plt.figure()
    ax_asec = SubplotHost(fig7, 1, 1, 1, aspect='auto')
    fig7.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('12co_26asec.m0.fits','12co_2asec.integrated.m0err.fits','ch3oh_26asec.m0.fits','ch3oh_masked.m0err.fits',restfreq_12co,restfreq_ch3oh,'12CO','CH3OH', 'C0')
    plotme(myDir+'annuli_12CO_CH3OH.txt', '', 'k', False)
    plotme(myDir+'pvratios_12CO_CH3OH.txt', '', 'k', False)
    plt.ylim(10,400)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    
    fig8 = plt.figure()
    ax_asec = SubplotHost(fig8, 1, 1, 1, aspect='auto')
    fig8.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('12co_26asec.m0.fits','12co_2asec.integrated.m0err.fits','hnco_26asec.m0.fits','hnco_masked.m0err.fits',restfreq_12co,restfreq_hnco,'12CO','HNCO', 'C0')
    plotme(myDir+'annuli_12CO_HNCO.txt', '', 'k', False)
    plotme(myDir+'pvratios_12CO_HNCO.txt', '', 'k', False)
    ratio, ratioerr = calcmomratios('12co_26asec.m0.fits','12co_2asec.integrated.m0err.fits','hco+_26asec.m0.fits','hco+_masked.m0err.fits',restfreq_12co,restfreq_hcop,'12CO','HCO+', 'C1')
    plotme(myDir+'annuli_12CO_HCO+.txt', '', 'k', False)
    plt.ylim(10,300)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    textcolors=['dodgerblue','orange']
    leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
    for i in enumerate(leg.get_texts()):
         i[1].set_color(textcolors[i[0]])
    
    fig9 = plt.figure()
    ax_asec = SubplotHost(fig9, 1, 1, 1, aspect='auto')
    fig9.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('cn32_26asec.m0v1.fits', 'cn32_2asec.m0err.fits', 'cs_26asec.m0.fits', 'cs_masked.m0err.fits', restfreq_cn32, restfreq_cs, 'CN32','CS','C8')
    plotme(myDir+'pvratios_CN32_CS.txt', '', 'k', False)
    plotme(myDir+'annuli_CN32_CS.txt', '', 'k', False)
    ratio, ratioerr = calcmomratios('cn32_26asec.m0v1.fits', 'cn32_2asec.m0err.fits', 'hcn_26asec.m0.fits', 'hcn_masked.m0err.fits', restfreq_cn32, restfreq_hcn, 'CN32','HCN','C9')
    plotme(myDir+'annuli_CN32_HCN.txt', '', 'k', False)
    plotme(myDir+'pvratios_CN32_HCN.txt', '', 'k', False)
    plt.xlim(0,15)
    plt.ylim(0.3,7)
    tedious_repetitive('Ratio')
    textcolors=['yellowgreen','xkcd:turquoise']
    leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
    for i in enumerate(leg.get_texts()):
         i[1].set_color(textcolors[i[0]])
         
    
    fig10 = plt.figure()
    ax_asec = SubplotHost(fig10, 1, 1, 1, aspect='auto')
    fig10.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('ch3oh_26asec.m0.fits','ch3oh_masked.m0err.fits','cs_26asec.m0.fits', 'cs_masked.m0err.fits',restfreq_ch3oh,restfreq_cs,'CH3OH','CS','C0')
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    
    fig11 = plt.figure()
    ax_asec = SubplotHost(fig11, 1, 1, 1, aspect='auto')
    fig11.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('hcn_masked.m0.fits','hcn_masked.m0err.fits','hnc_masked.m0.fits','hnc_masked.m0err.fits',restfreq_hcn,restfreq_hnc,'HCN','HNC', 'C1')
    plotme(myDir+'annuli_HCN_HNC.txt', '', 'blue', False)
    plotme(myDir+'pvratios_HCN_HNC.txt', '', 'blue', False)
    ratio, ratioerr = calcmomratios('cn32_na.mom0.fits','cn32_na.m0err.fits','cn12_na.mom0.fits','cn12_na.m0err.fits',restfreq_cn32,restfreq_cn12,'CN(3/2-1/2)','CN(1/2-1/2)', 'C4')
    plotme(myDir+'pvratios_CN32_CN12.txt', '', 'k', False)
    plotme(myDir+'annuli_CN32_CN12.txt', '', 'k', False)
    plt.axhline(2.0, linestyle=':', color='k')
    ratio, ratioerr = calcmomratios('12co21_sm_contsub.m0.fits','12co21_sm.m0err.fits','12co_hires_round_for21.m0.fits', '12co_hires_round_for21.m0err.fits',restfreq_12co21,restfreq_12co,'$^{12}$CO(2-1)','(1-0)','C2')
    plotme(myDir+'pvratios_12CO21_10_contsub.txt', '', 'k', False)
    plotme(myDir+'annuli_12CO21contsub_12CO.txt', '', 'k', False)
    plt.ylim(0.13,6)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    textcolors=['C1','C4','C2']
    leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
    for i in enumerate(leg.get_texts()):
         i[1].set_color(textcolors[i[0]])
    if (dosbprof):
        scalefactor = 5.5/np.nanmax(sbprof['sb'])
        plt.plot(sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey')
        plt.plot(-sbprof['off'],scalefactor*sbprof['sb'], color='darkgrey') # arbitrary scale factor to get the lines to show up on the plot
    if (dovline):
        plt.axvline(vloc, color='k', linestyle=':')
    fig11.savefig('scatterratios_more.pdf')
    
    fig12 = plt.figure()
    ax_asec = SubplotHost(fig12, 1, 1, 1, aspect='auto')
    fig12.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('hcn_26asec.m0.fits','hcn_masked.m0err.fits','hco+_26asec.m0.fits','hco+_masked.m0err.fits',restfreq_hcn,restfreq_hcop,'HCN','HCO+', 'C4')
    plotme(myDir+'annuli_HCN_HCO+.txt', '', 'k', False)
    plotme(myDir+'pvratios_HCN_HCO+.txt', '', 'k', False)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    
    # ------------------------
    # resolved CS/HCN
    fig13 = plt.figure()
    ax_asec = SubplotHost(fig13, 1, 1, 1, aspect='auto')
    fig13.add_subplot(ax_asec)
    ratio, ratioerr = calcmomratios('hcn_26asec.m0.fits', 'hcn_masked.m0err.fits', 'cs_26asec.m0.fits', 'cs_masked.m0err.fits', restfreq_hcn, restfreq_cs, 'HCN','CS', 'C9')
    plotme(myDir+'annuli_HCN_CS.txt', '', 'k', False)
    plotme(myDir+'pvratios_HCN_CS.txt', '', 'k', False)
    plt.xlim(0,15)
    tedious_repetitive('Ratio')
    
    
    # oiii/hb vs CS/HCN. not radial, just point vs point
    cs = fits.open(myDir+'cs_26asec.m0.fits')[0].data
    cs_err = fits.open(myDir+'cs_masked.m0err.fits')[0].data
    hcn = fits.open(myDir+'hcn_26asec.m0.fits')[0].data
    hcn_err = fits.open(myDir+'hcn_masked.m0err.fits')[0].data
    ratio = cs/hcn  # you should do the frequency correction thing
    ratioerr = ratio * np.sqrt((cs_err/cs)**2 + (hcn_err/hcn)**2)
    ratiosn = ratio/ratioerr
    useme = (cs/cs_err > 1.0) * (hcn/hcn_err > 1.0)
    minsn = np.min(ratiosn[useme])
    maxsn = np.max(ratiosn[useme])
    alphascale = (ratiosn-minsn)/(maxsn-minsn)
    oiiihb = fits.open(myDir+'NGC4526_logOIIIHB.regrid.fits')[0].data
    assert ratio.shape == oiiihb.shape, 'Are these the same region?'
    
    plt.figure()
    cmap = colors.LinearSegmentedColormap.from_list('incr_alpha', [(0, (*colors.to_rgb('C1'),0)), (1, 'C1')])
    plt.scatter(oiiihb[useme], ratio[useme], c=alphascale[useme], cmap=cmap, ec=None, s=30)    
    plt.plot([-0.5,0.1],[3.0,0.2], 'k') # eyeballing a line through the points in Tim's paper
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('log OIII/Hb')
    plt.ylabel('CS/HCN')
    
    
    # oiii/hb vs CN/HCN.  not radial, just point vs point
    cn = fits.open(myDir+'cn32_26asec.m0v1.fits')[0].data
    cn_err = fits.open(myDir+'cn32_2asec.m0err.fits')[0].data
    ratio = cn/hcn  # you should do the frequency correction thing
    ratioerr = ratio * np.sqrt((cn_err/cn)**2 + (hcn_err/hcn)**2)
    ratiosn = ratio/ratioerr
    useme = (cn/cn_err > 1.0) * (hcn/hcn_err > 1.0)
    minsn = np.min(ratiosn[useme])
    maxsn = np.max(ratiosn[useme])
    alphascale = (ratiosn-minsn)/(maxsn-minsn)
    assert ratio.shape == oiiihb.shape, 'Are these the same region?'
    
    plt.figure()
    cmap = colors.LinearSegmentedColormap.from_list('incr_alpha', [(0, (*colors.to_rgb('C1'),0)), (1, 'C4')])
    plt.scatter(oiiihb[useme], ratio[useme], c=alphascale[useme], cmap=cmap, ec=None, s=30)    
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('log OIII/Hb')
    plt.ylabel('CN/HCN')



#   CO vs dust continuum.
# for dust image, use 'projects/alma4526/cont_mfs_2_image_tt0.fits' or there's a new one that's better now

fig14 = plt.figure()
ax_asec = SubplotHost(fig14, 1, 1, 1, aspect='auto')
fig14.add_subplot(ax_asec)
# note on these function calls.  don't put in the real frequency of the continuum image because
# that will goof up some of the calculations.  it is hard-coded in the function above.
ratio, ratioerr = calcmomratios('cont_bigmfs_2asec_2t.im.pbcor.fits', 'cont_bigmfs_2asec_2t.im.pbcor.fits', 'c18o_masked.m0.fits','c18o_masked.m0err.fits', restfreq_c18o, restfreq_c18o, 'dust', 'C$^{18}$O', 'C2')
ratio, ratioerr = calcmomratios('cont_bigmfs_2asec_2t.im.pbcor.fits', 'cont_bigmfs_2asec_2t.im.pbcor.fits', '13co_masked.m0.fits','13co_masked.m0err.fits',  restfreq_13co, restfreq_13co, 'dust', '$^{13}$CO', 'C1')
ratio, ratioerr = calcmomratios('cont_bigmfs_2asec_2t.im.pbcor.fits', 'cont_bigmfs_2asec_2t.im.pbcor.fits', '12co_2asec.integrated.fits','12co_2asec.integrated.m0err.fits', restfreq_12co, restfreq_12co, 'dust', '$^{12}$CO', 'C0')
plt.axhline(4.3, color='C0', linestyle=':')
plt.text(0.7, 4.8, 'MW')
plt.xlim(0,15)
plt.ylim(2,500)
tedious_repetitive('$\\alpha_{X}~~$ ($\mathrm{M}_\odot$ pc$^{-2}$ / (K km s$^{-1}$))')
textcolors=['C2','C1','C0']
leg = ax_asec.legend(loc='best', frameon=False, handlelength=0, scatterpoints=0)
for i in enumerate(leg.get_texts()):
     i[1].set_color(textcolors[i[0]])
fig14.savefig('alphas.pdf')