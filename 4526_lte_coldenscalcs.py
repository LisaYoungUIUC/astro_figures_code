#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:56:34 2021

@author: lyoung

PURPOSE: do the column density calculations for the various species in 4526.  
Based off strategies developed in coldens_calcs.py but that one was for 7465 so the input strategy is different
and now I'm doing more than one position, etc.
Reads the mom0 images, picks out the peak, and prints LTE column density estimate.
Also uses the pv slices for some radial plots of surface brightness. (this was an attempt to see if any
species has obviously different radial behavior.)

04jun2021 status.  this makes two figures, one with raw LTE column density estimates for each species
and another that scales the column densities by their assumed abundances to infer N(H2).
If the assumed abundances are correct (and the lines are optically thin), in theory they might all say
the same thing about N(H2).
Raw abundances from DMeier had 13co/c18o = 250/60 = 4.2.  This set of numbers gives NH2 inferred from 13co
to be a bit higher than NH2 inferred from c18o.
Does that make sense?  cause I also have the line ratios themselves which, at least in the outer parts of the
galaxy, require a somewhat higher 13co/c18o ratio more like 7 to 9.  so less c18o.
optical depth effects might also be important and then 13co (and 13co-derived NH2) might be underestimates
of the true values.

28aug2021 status.  this now runs.  It reads off peak of mom0 in every detected line and computes the
associated column density for LTE @ 20K or whatever other temp you choose.  Prints out resolution, peak
value in Jy/b*km/s and column density assuming some optical depth.  I put the optical depth correction in
because I claim I actually know it for CN; for other lines we have loose to no constraints so you will have
to look at those assumptions carefully when constructing the table.
It also attempts to plot some radial SB profiles that are made by squashing the pv slices.  
The squashed pv slices are made by 4526pvslicefig.py but I haven't done them all so in fact most aren't
correct.  I had thought maybe the radial line ratios would be obvious.  In fact the trends in 12co/c18o 
and 12co/cn are obvious but I'm not sure that teaches me anything I didn't know before.  Might go ahead
and finish the rest of the squashes just in case anything leaps out.  but then probably just a table
in the paper with the mom0 values, and then people will (as usual) have to keep in mind about the optical
depths, tho I could compare the CO ones with radex values.

ch3oh seems to have a stronger nuclear peak than most?  well, not clear what to make of that since the 
ch3oh is really faint, mom0 is kinda crap looking, hard to tell what's going on.'
why does HNCO have such a high column density?  that is weird.  does it happen to me (7465) and
dave?  well, 7465 just had a limit on it but the column densities have N(HNCO) > N(HCN) in several places in
both Meier 2015 and Meier 2012, so it seems like that might not be so weird. chemistry-wise, still
not sure what is going on.  there are probably some optical depth effects happening as well; HCN might
have higher optical depth.

assumptions on tau13 and tau12?  set them all at 0.01 even tho I'm pretty sure it might be as large 
       as 10 for 12co1-0.
       
Jan 2022.  Dave makes a couple of comments - first, he wants beam sizes the same, and second, he thinks that
I should not correct CN for optical depth, or maybe should do that in the comments since I can't do it for
all molecules similarly.  Also putting in two temperatures.

And doing simple theoretical calcs for comparing optical depths of HCN and CN assuming typical abundances.
 
12 Feb 2022 upgraded to Mangum&Shirley eq 80 rather than 82.  It might make a slight difference.
   
TO DO:
    - if you want to get a little more precise with this, you could read off the mom0 value towards the
        nucleus rather than just grabbing the max.  It might make a slight difference in some cases.
    - should probably fix the print statements and also dump the modified ch3oh calculations to the output file.
    - if you like, you can put the print statement for the pv slices back in (sanity check?)

DONE:
    - finish sanity checks on units
    - enhance this so you can do 13co and c18o as well as CN
    - probably should upgrade to Mangum&Shirley eq 80 rather than 82.  It might make a slight difference.

"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
from astropy.io import fits
import datetime

# where to find files & stuff
prefix = '/Users/lyoung/' if os.path.isdir('/Users/lyoung') else '/Users/lisayoung/'
myDir = prefix+'Dropbox/projects/alma4526/'

# constants, cgs
h = np.double(6.62e-27) # Planck  # actually Python's default calculations are done in double, so not necessary to force it
c = np.double(3.0e10) # light speed
kb = np.double(1.38e-16) # Boltzmann


def getlineparams(species,transID,upID):
    logtemp_ary = np.arange(-1.,3.0,step=0.01)
    temp_ary = 10.**logtemp_ary
    filename = '/Users/lyoung/software/Radex/data/'+species+'.dat'
    # get the count of energy levels in the file (helps with logistics of reading the file)
    datafile = open(filename)
    for i in range(6):
        line = datafile.readline() # skip 1st 5 lines; 6th has the info we want
    nlevs = int(line)
    # print(nlevs,' lines')
    for i in range(nlevs+3):
        line = datafile.readline() # skip all these
    ntrans = int(line)
    datafile.close()
    # top part of file tabulates energy levels and their degeneracies
    data = np.genfromtxt(filename, dtype=None, usecols=[0,1,2], names=['level','energy','weight'],\
                         skip_header=7, max_rows=nlevs)
    useme = (data['level'] == upID)
    gup = data['weight'][useme][0]
    energy_K = data['energy'] * h * c / kb # convert to K for the energy levels
    if (species == 'hcn'):
        # diagnostic printout
        print(data['energy'][0:5])
        print(energy_K[0:5])
    Eup = energy_K[useme][0]
    nlevs = len(data)    
    x_ary = np.zeros((nlevs, len(temp_ary)))
    for i in range(nlevs):
        x_ary[i,:] = (data['weight'][i]/data['weight'][0]) * np.exp(-((energy_K[i]-energy_K[0])/temp_ary))
    # sum over all levels to give the partition fn (will be an array, a fn of temp)
    part_fn = np.sum(x_ary, axis=0)
    # normalize level pops to get fractional level pops
    # x_ary = x_ary / part_fn  # don't need this in current application, but it is here for reference
    # part_fn appears to be the thing called Q or Qrot in, for example, the radex paper.
    #
    # next part of file tabulates Einstin As for the transitions
    data2 = np.genfromtxt(filename, dtype=None, usecols=[0,3,4], names=['trans','Einsteins','freqs'], \
                          skip_header=7+nlevs+3, max_rows=ntrans)
    useme2 = (data2['trans'] == transID)
    Aul = data2['Einsteins'][useme2][0]
    freq = data2['freqs'][useme2][0]  # this is just a sanity check to make sure I've identified the correct transition
    # note here freq is in units of GHz and Eup is in K.
    return part_fn, temp_ary, Aul, Eup, gup, freq

def Jnu(freq,temp):
    jnu = (h * freq * 1.0e9 / kb) / (np.exp(h*freq*1.0e9/(kb*temp)) - 1.)
    return jnu

def taucalcs(species,transID,upID,abun):
    # This is based on Mangum & Shirley eq 32
    coldens = 1.e22 * abun  # taking a typical N(H2) about 1.e22
    qrot_ary, temp_ary, Aul, Eup, gup, freq = getlineparams(species, transID, upID)
    print(species, 'Trans %i up %i Aul=%.2e, Eup=%.2f, gup=%i, freq=%.2f'%(transID, upID, Aul, Eup, gup, freq))
    part5 = Aul * c**3 / (8. * np.pi * freq**3 * 1.e27)  # the 1e27 is because freq is in GHz
    part6 = gup / qrot_ary 
    part7 = np.exp(-Eup/temp_ary)
    part8 = np.exp(h * freq * 1.e9 / (kb * temp_ary)) - 1.
    tau = (coldens/dv) * part5 * part6 * part7 * part8
    return tau
    

# ------------------  work starts here ------------------------
    
temp = 20 # K
temp2 = 10 # K .  Dave wants a lower value
tbg = 2.73 # K.  CMB, for background.
print('Tex = %.1f K'%(temp))
doscaled = False # make another plot that uses assumed abundance to infer H2 
dohires = False # Use the mom0s at their nominal resolution for the coldens calcs.  otherwise does 2.6" for all.  Doesn't affect the business with the pv slices.
dotaucorrs = False # correct the column densities for tau ~ 1.  I don't have tau estimates for most molecules.

species_list = ['cn','13co','c18o','12co','cs','ch3oh','hnc','hco+','hcn','hnco43']
label_list = ['CN','\\thirco', '\\ceighto','\\tweco','CS','\\chhhoh','HNC','\\hcop','HCN','HNCO']
prof_list = ['SBprof_CN32.txt','SBprof_13CO.txt','SBprof_C18O.txt','SBprof_12CO.txt','SBprof_CS.txt',\
             'SBprof_CH3OH.txt','SBprof_HNC.txt','SBprof_HCO+.txt','SBprof_HCN.txt','SBprof_HNCO.txt']
pv_list = ['pv_cn32_2asec_15kms.fits','pv_13co_15kms_morechans.fits','pv_c18o_15kms_morechans.fits', \
           'pv_12co_2asec_15kms.fits','pv_cs_sm.fits','pv_ch3oh_sm.fits','pv_hnc_sm.fits',\
               'pv_hco+_sm.fits','pv_hcn_sm.fits','pv_hnco.fits']
if (dohires):  # pick this one if you don't care about matched resolution
    mom0_list = ['cn32_na.mom0.fits','13co_masked.m0.fits','c18o_masked.m0.fits','12co_hires.integrated.fits','cs_masked.m0.fits',\
                 'ch3oh_masked.m0.fits','hnc_masked.m0.fits','hco+_masked.m0.fits','hcn_masked.m0.fits','hnco_masked.m0.fits']  
    # pick this one if you need matched resolution for some of these species
    # mom0_list = ['cn32_2asec.mom0.fits','13co_masked.m0.fits','c18o_masked.m0.fits'] 
    mom0err_list = ['cn32_2asec.m0err.fits','13co_masked.m0err.fits','c18o_masked.m0err.fits','12co_hires_integrated_err.fits','cs_masked.m0err.fits',\
                 'ch3oh_masked.m0err.fits','hnc_masked.m0err.fits','hco+_masked.m0err.fits','hcn_masked.m0err.fits','hnco_masked.m0err.fits']
else:  # everything at matched 2.6" resolution
    mom0_list = ['cn32_26asec.m0v2.fits','13co_26asec.m0.fits','c18o_26asec.m0.fits','12co_26asec.m0.fits','cs_26asec.m0.fits',\
                 'ch3oh_26asec.m0.fits','hnc_26asec.m0.fits','hco+_26asec.m0.fits','hcn_26asec.m0.fits','hnco_26asec.m0.fits']
    # these mom0errs are not all at 2.6" res; some are at 2.0", and some are 2 < x < 2.6; but they are pretty close. so I'm using them.  in most cases that will overestimate the error a little because the peaks are higher at hi res.
    mom0err_list = ['cn32_2asec.m0err.fits','13co_masked.m0err.fits','c18o_masked.m0err.fits','12co_2asec.integrated.m0err.fits','cs_masked.m0err.fits',\
                 'ch3oh_masked.m0err.fits','hnc_masked.m0err.fits','hco+_masked.m0err.fits','hcn_masked.m0err.fits','hnco_masked.m0err.fits']
if (dotaucorrs):
    tau_list = [4.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]  # these are estimated optical depths in the nucleus 
else:
    tau_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]  # low value << 1 makes effectively no correction
est_12co13co = 6.5  # these are just for scaling, in case you want to look for differences between the species in the visual plot
est_13coc18o = 5.8
# maybe just fill out abun_list with 1s if don't care about scaling them
abun_list = [3.e-8, 8.e-5/est_12co13co, 8.e-5/(est_12co13co*est_13coc18o),1,1,1,1,1,1,1]  # abundances (relative to H2) copied from DMeier, 2015 for CN and 2008 for CO

# output latex table with LTE column densities
if (dohires):
    outfile = open('lte_coldenscalcs_hires.tex', 'w')
    outfile.write("% {:%Y-%b-%d %H:%M:%S} by 4526_lte_coldenscalcs.py\n".format(datetime.datetime.now()))
    outfile.write('% species & beam & peak mom0 (Jy/b*km/s) & coldens \n')
else:
    outfile = open('lte_coldenscalcs_lowres.tex', 'w')
    outfile.write("% {:%Y-%b-%d %H:%M:%S} by 4526_lte_coldenscalcs.py\n".format(datetime.datetime.now()))
    outfile.write('% species & peak mom0 (Jy/b*km/s) & coldens1 & coldens2 \n')

plt.figure('notscaled')
if (doscaled):
    plt.figure('scaled')

for i in range(len(species_list)):

    species = species_list[i]
    label = label_list[i] # for dumping to latex table
    abun = abun_list[i]
    tau = tau_list[i]
    sbdata = np.genfromtxt(myDir+prof_list[i], usecols=[0,1,2], names=['offset','sb','sb_unc'],\
                          dtype=None)
    sbdata['sb'] *= 1.0e5  # now Jy*cm/s
    sbdata['sb_unc'] *= 1.0e5  # now Jy*cm/s
    
    pvfile = fits.open(myDir+pv_list[i])
    pvheader = pvfile[0].header
    pv_fwhm1 = pvheader['bmaj']
    pv_fwhm2 = pvheader['bmin'] # these are probably in degrees
    beamarea_deg = 1.1331*pv_fwhm1*pv_fwhm2
    pv_beamarea_sr = beamarea_deg * (np.pi/180.)**2
    pv_fwhm1 *= 3600. # now in arcsec for printout
    pv_fwhm2 *= 3600.
    
    m0file = fits.open(myDir+mom0_list[i])
    m0header = m0file[0].header
    m0_fwhm1 = m0header['bmaj']
    m0_fwhm2 = m0header['bmin'] # these are probably in degrees
    m0data = m0file[0].data
    m0max = np.nanmax(m0data)
    m0err = fits.open(myDir+mom0err_list[i])[0].data
    m0max_unc = m0err[np.where(m0data == m0max)][0]
    m0_beamarea_sr = 1.1331 * m0_fwhm1 * m0_fwhm2 * (np.pi/180.)**2
    m0_fwhm1 *= 3600. # now in arcsec for printout
    m0_fwhm2 *= 3600.
    
    
    #  be a little careful about which transition we're working with
    if (species == 'cs'):
        transID = 2 # I have J=2-1
        upID = 3 # ID for the upper level of the transition (top part of the data file.  confusingly these start at 1 whereas J starts at 0.)
    elif (species == 'cn'):
        transID = 2 # here I'm doing only the 113.49 GHz transition. but I think it has helpfully summed the internal fine structure for me
        upID = 3
    elif (species == 'cch'):
        # this one is very complicated, and the data file breaks the fine structure out individually which
        # is not what I'm doing.  I'm using the sum of the group at 87.3 GHz.  so will, for now, cheesily
        # adopt one set of values; maybe the one with the highest degeneracy and Einstein A
        transID = 3
        upID = 4
    elif (species == 'ch3oh'):
        # similar deal to cch
        # alma OT says these are vt=0, 2(x,y)-1(u,v) for xyuv = -1 to 2. at about 96.73 GHz rest.
        # this is actually 3 transitions in the radex data file, transID = 86, 87, 88. with
        # level IDs (in the radex list) 2-1, 6-4 and 9-7.  pretty similar einstein As.  plus another one in A.
        # and those transitions all have statistical weights of 5 upper and 3 lower but different 
        # energies (one set is at 12K, one at 20 K, one at 28K)
        # the one at 20 K has the highest Einstein A. hm. done more carefully below.
        transID = 87
        upID = 6
    elif (species == 'sio'):
        transID = 2
        upID = 3
    elif (species == 'hnco43'):
        transID = 37
        upID = 5
    else:
        transID = 1 # these are most cases with the J=1-0 transition
        upID = 2
    # get all the necessary info for this line
    qrot_ary, temp_ary, Aul, Eup, gup, freq = getlineparams(species, transID, upID)
    qrot_interp = np.interp(temp, temp_ary, qrot_ary) # this value is specific for the chosen temperature
    qrot_interp_2 = np.interp(temp2, temp_ary, qrot_ary) # Dave likes a lower T for the HD tracers
    if (species == 'c18o'):
        print('Qrot_interp', qrot_interp)
    
    # this calculation is based on Mangum & Shirley eq 80.  without part3b it is eq 82.
    part1 = 4.*np.pi / (Aul * h * c)
    part2 = qrot_interp / gup
    part2_2 = qrot_interp_2 / gup
    part3 = np.exp(Eup/temp)
    part3_2 = np.exp(Eup/temp2)
    part3b = Jnu(freq,temp) / (Jnu(freq,temp) - Jnu(freq,tbg))
    part3b_2 = Jnu(freq,temp2) / (Jnu(freq,temp2) - Jnu(freq,tbg))
    part4 = 1.0e-23 / pv_beamarea_sr  # there's also a factor of 10^5 for cm/km, and it's above
    pv_convfactors = part1 * part2 * part3 * part3b * part4
    m0_convfactors = part1 * part2 * part3 * part3b * 1.0e-23 * 1.0e5 / m0_beamarea_sr # doing this twice is in case m0 and pv have different beam sizes
    m0_convfactors_2 = part1 * part2_2 * part3_2 * part3b_2 * 1.0e-23 * 1.0e5 / m0_beamarea_sr # Dave likes a lower T for the HD tracers

    # just reading peak value off mom0
    m0max_coldens = m0max * m0_convfactors * tau / (1. - np.exp(-tau))
    m0max_coldens_unc = m0max_coldens * (m0max_unc/m0max) # m0max_unc * m0_convfactors * tau / (1. - np.exp(-tau))
    logcoldens = np.log10(m0max_coldens)
    logcoldensunc1 = np.log10(1.0+(m0max_coldens_unc/m0max_coldens))
    logcoldensunc2 = -1.*np.log10(1.0-(m0max_coldens_unc/m0max_coldens)) # forcing this one positive for printing purposes
    m0max_coldens_2 = m0max * m0_convfactors_2 * tau / (1. - np.exp(-tau))  # Dave likes a lower T for the HD tracers
    m0max_coldens_unc_2 = m0max_coldens_2 * (m0max_unc/m0max) 
    logcoldens_2 = np.log10(m0max_coldens_2)  # other temp
    logcoldensunc1_2 = np.log10(1.0+(m0max_coldens_unc_2/m0max_coldens_2))
    logcoldensunc2_2 = -1.*np.log10(1.0-(m0max_coldens_unc_2/m0max_coldens_2)) # forcing this one positive for printing purposes
    print('%s peak N from m0 at %.2f x %.2f" is %.2f +- %.2f Jy/b*km/s or %.2e +- %.2e'%(species,m0_fwhm1,m0_fwhm2,m0max,m0max_unc,m0max_coldens,m0max_coldens_unc))
    if (dohires):
        outfile.write('%s & %.1f $\\times$ %.1f &  %.2f $\\pm$ %.2f & %.2f $\pm$ %.2f \\\\\n'%(label,m0_fwhm1,m0_fwhm2,m0max,m0max_unc,logcoldens,logcoldensunc2))
    else:  # don't need to write out beam, they're all the same. but now writing out 2 temps.
        if (species=='12co') or (species=='13co'):
            # more digits on log N bc uncertainty is so small
            outfile.write('%s &  %.2f $\\pm$ %.2f & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f \\\\\n'%(label,m0max,m0max_unc,logcoldens,logcoldensunc2,logcoldens_2,logcoldensunc2_2))
        else:
            # 2 digits are enough on log N
            outfile.write('%s &  %.2f $\\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f \\\\\n'%(label,m0max,m0max_unc,logcoldens,logcoldensunc2,logcoldens_2,logcoldensunc2_2))

    # now plotting surface density from slices along the major axis
    coldens = pv_convfactors * sbdata['sb']
    coldens_unc = pv_convfactors * sbdata['sb_unc']
    spaxwithdata = (coldens >= 0.)
    # optical depth correction (where known)
    corr_coldens = coldens * tau / (1. - np.exp(-tau)) # strictly speaking this is only accurate in the nucleus because that's where the tau is measured
    corr_coldens_unc = coldens_unc * tau / (1. - np.exp(-tau))
    index_ctr = np.where(sbdata['offset'] == np.min(np.abs(sbdata['offset'])))
    # scaling by assumed abundance to get H2.  You may not care about this.
    esth2 = coldens/abun
    esth2_unc = coldens_unc/abun
    corr_esth2 = esth2 * tau / (1. - np.exp(-tau)) # strictly speaking this is only accurate in the nucleus because that's where the tau is measured

    plt.figure('notscaled')
    plt.errorbar(sbdata['offset'][spaxwithdata], coldens[spaxwithdata], yerr=coldens_unc[spaxwithdata], label=species.upper())
    plt.errorbar(sbdata['offset'][index_ctr], corr_coldens[index_ctr], yerr=corr_coldens_unc[index_ctr], fmt='ro') # correction for CN and 13co column densities being not very optically thin    
    # diagnostic print statement was mostly as a sanity check on calcs using mom0 vs slice
    # calcs from slices are a little bit lower than from peak of mom0 (that's averaging) and they have larger uncertainties
    # print('%s peak N from slice at %.1f x %.1f" is %.2f +- %.2f Jy/b*km/s or %.2e +- %.2e'%(species,\
         #    pv_fwhm1,pv_fwhm2,(sbdata['sb'][index_ctr]/1.0e5),(sbdata['sb_unc'][index_ctr]/1.0e5),corr_coldens[index_ctr],corr_coldens_unc[index_ctr]))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.axvline(6.0, linestyle='--')
    plt.axvline(-6.0, linestyle='--')
    plt.xlabel('Offset (")')
    plt.ylabel('Coldens (cm$^{-2}$)')
    plt.legend()
    
    if (doscaled):
        plt.figure('scaled')
        plt.errorbar(sbdata['offset'][spaxwithdata], esth2[spaxwithdata], yerr=esth2_unc[spaxwithdata], label=species.upper())
        plt.errorbar(sbdata['offset'][index_ctr], corr_esth2[index_ctr], fmt='ro') # correction for CN and 13co column densities being not very optically thin    
        ax = plt.gca()
        ax.set_yscale('log')
        plt.axvline(6.0, linestyle='--')
        plt.axvline(-6.0, linestyle='--')
        plt.xlabel('Offset (")')
        plt.ylabel('Coldens (cm$^{-2}$)')
        plt.legend()
        
outfile.close()

#%%

#   ch3oh needs more help and more care, because the observed line is made up of several components
# the components are in the radex data files as
# e-ch3oh.dat 
# transition 86, upper level = 2, lower level = 1
# transition 87, upper level = 6, lower level = 4
# transition 88, upper level = 9, lower level = 7
# ch3oh_a.dat
# transition 56, upper level = 3, lower level = 2
print('More detailed ch3oh calculations')

species_list = ['e-ch3oh','e-ch3oh','e-ch3oh','ch3oh_a']
transID_list = [86, 87, 88, 56]
upID_list = [2, 6, 9, 3]
stuff_ary = np.zeros_like(temp_ary)
stuff_ary_82 = np.zeros_like(temp_ary)


for species,transID,upID in zip(species_list,transID_list,upID_list):
    qrot_ary, temp_ary, Aul, Eup, gup, freq = getlineparams(species, transID, upID)
    print('trans Aul Eup gup freq',transID, Aul, Eup, gup, freq)
    part9 = Aul * h * c**3 / (8.*np.pi * kb * freq**2 * 1.e18)  # the 1.e18 is because freq is in GHz
    part10 = gup / qrot_ary
    part11 = np.exp(-Eup/temp_ary)
    part12 = (Jnu(freq, temp_ary) - Jnu(freq, 2.73)) / Jnu(freq, temp_ary)  # 2.73 is obviously CMB
    stuff_ary_82 += part9 * part10 * part11
    stuff_ary += part9 * part10 * part11 * part12
    # without part 12 it is Mangum & Shirley eq 82.  With part 12, it is eq 80.
    
plt.figure()
plt.loglog(temp_ary, stuff_ary)
plt.loglog(temp_ary, stuff_ary_82)
plt.xlabel('log Temp (K)')
plt.ylabel('stuff (= line intensity / N(ch3oh)')

plt.figure()
plt.loglog(temp_ary, part12)

stuff1 = np.interp(10., temp_ary, stuff_ary)
stuff2 = np.interp(20., temp_ary, stuff_ary)
stuff3 = np.interp(200., temp_ary, stuff_ary)
# observed line intensity is 4.35 K km/s see p 147
line = 4.35 # K km/s
cd1 = line*1.e5 / stuff1  # the 1.e5 is to convert to cm/s to match other cgs calculations
cd2 = line*1.e5 / stuff2
cd3 = line*1.e5 / stuff3
print('for %.2f K km/s and 10 K I find log N(ch3oh) = %.2f'%(line,np.log10(cd1)))
print('for %.2f K km/s and 20 K I find log N(ch3oh) = %.2f'%(line,np.log10(cd2)))
print('for %.2f K km/s and 200 K I find log N(ch3oh) = %.2f'%(line,np.log10(cd3)))
outfile = open('lte_coldenscalcs_lowres.tex', 'w')
# ### THIS write statement is not correct but it's a reminder to me that I should fix it up and write out these new calculations
outfile.write('%s &  %.2f $\\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f \\\\\n'%('ch3oh',m0max,m0max_unc,logcoldens,logcoldensunc2,logcoldens_2,logcoldensunc2_2))
outfile.close()


#%%
# now see what we can do with the dust?
    
# lifting some calculation stuff from the 4526_scatter_lineratios.py where I have taken the 
# dust image out to Msun/pc2 and N(H2).
# careful here - some of these constants are redefined with different units from above
h = 6.62e-34 # J s
c = 3.0e8 # m/s
kb = 1.38e-23 # J/K
gasdust = 100 # this is m(H2)/m(dust) -- not including He
doAuld = False# which MBB dust model?
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
nu_cont = 99.33e9 # Hz.  this is for cont_bigmfs_2asec_2t
lam_cont = c/nu_cont # in meters
BnuT = 2.* h * (nu_cont)**3 / (c**2 * (np.exp(h*nu_cont/(kb*Tdust))-1.)) # SI units, mks
opacity = opacity_coef * (1.e-6*scalewave/lam_cont)**Beta  

contfile = fits.open(myDir+'cont_slice.fits')
contslice = np.squeeze(contfile[0].data)
# cont_bigmfs_2asec_2t.im.pbcor has rms about 1.25e-5 Jy/b near the center
rms_orig = 1.25e-5 # in the raw continuum image, ignoring the averaging that went into making the slice because
# the slice is exactly 1 beam (5 pix) wide, so you don't gain much S/N in averaging that way
hdr = contfile[0].header
bmaj1 = hdr['BMAJ'] # degrees
bmin1 = hdr['BMIN']
beamarea_sr = 1.1331 * bmaj1 * bmin1 * (np.pi/180.)**2 
bmaj1 *= 3600. # now in arcsec
bmin1 *= 3600. 
factor1 = 1.e-26 / beamarea_sr # Jy/b to W/m2.Hz.sr
factor2 = gasdust / (BnuT * opacity) # W/m2.Hz.sr to kg/m2. 
factor3 = (3.09e16)**2 / 1.99e30 # kg/m2 to  Msun (of H2, not He) / pc2
factor4 = 6.24e19 # Msun/pc2 to molecules/cm2 (i.e. 6e19 mol/cm2 per Msun/pc2)
contslice *= factor1 * factor2 * factor3 * factor4 # now N(H2)
rms = rms_orig * factor1 * factor2 * factor3 * factor4 # now also N(H2), 1sigma


plt.figure()
# header params you want crval1, cdelt1, crpix1 and it's already in arcsec
contslice_offset = hdr['crval1'] + (np.arange(1,hdr['naxis1']+1) - hdr['crpix1']) * hdr['cdelt1']
plt.plot(contslice_offset, contslice)
plt.axhline(0.0)

plt.axhline(rms, linestyle='--', color='k')
plt.axhline(-rms, linestyle='--', color='k')
plt.axvline(1.3*bmaj1, linestyle='--', color='k')
plt.axvline(-1.3*bmaj1, linestyle='--', color='k')
plt.ylim(-1e22, 1.2e23 )
plt.ylabel('N(H2) cm$^{-2}$')
plt.xlabel('Offset (")')

#%%      
#        checking tau_HCN vs tau_CN
# copying lots of useful code from above.
# will do the calculations for a range of temps

# constants, cgs.
# careful, the constants were redefined for the dust business above.
# but getlineparams wants them cgs.
h = np.double(6.62e-27) # Planck  # actually Python's default calculations are done in double, so not necessary to force it
c = np.double(3.0e10) # light speed
kb = np.double(1.38e-16) # Boltzmann
species_list = ['cn','hcn']
transID_list = [2, 1] # see notes above
upID_list = [3, 2]
cnhcn_list = [1., 5., 0.2] # this is the abundance ratio [CN/HCN].  Dave says usually in the range of 1 to 5.
abuncn = 3.e-8 
dv = 30.e5 # cm/s, specific value doesn't matter, same linewidth for both species
tau_CN = taucalcs(species_list[0], transID_list[0], upID_list[0], abuncn)
plt.figure()
for cnhcn in cnhcn_list:
    tau_HCN = taucalcs(species_list[1], transID_list[1], upID_list[1], abuncn/cnhcn)
    ratio = tau_HCN/tau_CN
    plt.loglog(temp_ary, ratio, label=str(cnhcn))
plt.xlim(1.,100.)
plt.ylim(0.2,200.)
plt.xlabel('Temp (K)')
plt.ylabel('$\\tau(HCN)/\\tau(CN)$')
plt.legend(frameon=False)


species_list = ['cn','hcn','hco+','hnc','hnco','hnco']
transID_list = [2, 1, 1, 1, 37, 42] # see notes above
upID_list = [3, 2, 2, 2, 5, 6]
abuncn = 3.e-8
cnx = 1.  # this is [CN/X] for X = hcn, hco+, hnc
dv = 30.e5 # cm/s, specific value doesn't matter, same linewidth for both species
tau_CN = taucalcs(species_list[0], transID_list[0], upID_list[0], abuncn)
plt.figure()
for species,transID,upID in zip(species_list,transID_list,upID_list):
    tau_X = taucalcs(species, transID, upID, abuncn/cnx)
    ratio = tau_X/tau_CN
    plt.loglog(temp_ary, ratio, label='CN/'+species.upper())
plt.xlim(1.,100.)
plt.ylim(0.01,200.)
plt.xlabel('Temp (K)')
plt.ylabel('$\\tau(X)/\\tau(CN)$')
plt.legend(frameon=False)
