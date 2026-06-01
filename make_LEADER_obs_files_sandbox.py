#!/usr/bin/env python
# coding: utf-8

# # This code writes a curl script to query a defined WISE/NEOWISE database from IRSA.  It only queries asteroids in a user-defined collisional family.
# 
# After executing the resultant script, the next step is to filter and reformat the results into .obs files ready for ingest into the LEADER code
# 
# 
# Current moving-object enabled catalogs, per https://irsa.ipac.caltech.edu/onlinehelp/catalogs/#api: 
# 
# All-Sky: allsky_4band_p1bs_psd 
# 
# WISE 3band: allsky_3band_p1bs_psd 
# 
# WISE 2band: allsky_2band_p1bs_psd 
# 
# NEOWISE-R: neowiser_p1bs_psd

# In[336]:


famid = '350'                              ## Identifier for the asteroid collisional family that will be written into the curl script.
cat = 'allsky_4band_p1bs_psd'               ## which catalog you want to query
min_obs = 5                                 ## minimum number of observations required for a .obs file to be written
istart = 0                                 ## the index at which to start writing output .obs files (in case a previous execution was interrupted)
overwrite = False                           ## overwrite existing curl script and erase .tbl data? 
filterpriority = 'w3'                       ## filter you plan to analyze (lowercase)

fle = 'AllMBAFamilyMembers.txt'             ## path to file that lists asteroid family designations from 
                                            ## http://asteroids.matf.bg.ac.rs/fam/familymembers.php downloaded
                                            ## July 2025. Website said the list was last updated April 2023. They are
                                            ## concatenated with the newly identified family members in Nesvorny et al. 2024

#fle_m = 'MBAFamilies_Masiero13.txt'         ## path to another file that lists asteroid family designations from 
                                            ## Masiero et al. (2013).  As of August 8, 2025, this file is not used.
                                            ## It may be incorporated in future versions of this code to find family
                                            ## members that 

neowise_fle = 'neowise_mainbelt.csv'        ## name of the asteroid catalog file containing names and NEOWISE-determined properties, 
                                            ## from the PDS SBN; Mainzer et al. (2019)

#outfile = 'GetWiseData_FamID'+famid+'.sh'   ## name and path to curl script that will be written by this code


# In[337]:


import numpy as np
import matplotlib.pyplot as plt
import atpy
import sys
import shutil
import os
from math import *
import requests
from astropy.time import Time
from astropy import units as u
from sunpy.coordinates import get_horizons_coord
from astropy.coordinates import SkyCoord, CartesianRepresentation, get_sun


# In[338]:


## write the folder to which the curl script and IRSA tables will be downloaded

try:
    os.mkdir('Fam'+famid+'_data_'+cat+'_'+filterpriority)
except:
    if overwrite is True:
        shutil.rmtree('Fam'+famid+'_data_'+cat+'_'+filterpriority)
        os.mkdir('Fam'+famid+'_data_'+cat+'_'+filterpriority)
    else:
        print('Fam'+famid+'_data_'+cat+'_'+filterpriority+' directory already exists')

if filterpriority == 'w2':
    ifilt = 1
if filterpriority == 'w3':
    ifilt = 2



# In[339]:


## read in the file that records the family designations for MBAs from http://asteroids.matf.bg.ac.rs/fam/familymembers.php
famid_all, mpecobj_all, objid_all = np.genfromtxt(fle,unpack=True,dtype=str,usecols=(0,1,2))
objid = objid_all.compress((famid_all == famid).flat)
objid_mpec = mpecobj_all.compress((famid_all == famid).flat)

## the excerpt below is not used
### read in the masiero file as well
#objid_all_m,famid_all_m = np.genfromtxt(fle_m,unpack=True,dtype=str,usecols=(0,5))
#objid_m = objid_all_m.compress((famid_all_m == famid).flat)


# In[340]:


print('Number of asteroids in this family = '+str(len(objid)))


# In[127]:


## read the neowise/wise file of MBAs observed by the mission

objnum_n,provdesig_n,name_mpced_n,diam_n,diamerr_n = np.genfromtxt(neowise_fle,
                                                                   unpack=True,
                                                                   usecols=(0,1,2,11,12),
                                                                   delimiter=',',
                                                                   dtype=str
                                                                  )


# In[43]:


## in cases of more than one diameter determination, consolidate the arrays by selecting the entry 
## with the smallest diameter uncertainty

u_objnum_n, u_provdesig_n, u_name_mpced_n, u_name_curl_n, u_diam_n, u_diamerr_n = [], [], [], [], [], []

unique_mpecs_n = np.asarray(list(set(name_mpced_n)))
for i in range(len(unique_mpecs_n)):
    imatch = np.where(name_mpced_n == unique_mpecs_n[i])[0]

    if len(imatch) > 1:        
        ibest_group = np.where(diamerr_n[imatch] == min(diamerr_n[imatch]))[0]
        if len(ibest_group) > 1:
            ibest_group = ibest_group[0]
        ibest = imatch[ibest_group]
    elif len(imatch) == 1:
        ibest = imatch
    else:
        print('problem finding index matches for '+str(unique_mpec_n[i]))

    u_objnum_n.append(objnum_n[ibest][0])
    u_provdesig_n.append(provdesig_n[ibest][0].replace('"','').replace(' ',''))
    u_name_mpced_n.append(name_mpced_n[ibest][0].replace('"','').rstrip())
    u_diam_n.append(diam_n[ibest][0])
    u_diamerr_n.append(diamerr_n[ibest][0])

u_objnum_n = np.asarray(u_objnum_n)
u_provdesig_n = np.asarray(u_provdesig_n)
u_name_mpced_n = np.asarray(u_name_mpced_n)    
u_diam_n = np.asarray(u_diam_n)
u_diamerr_n = np.asarray(u_diamerr_n)



# In[341]:


## identify the object ids of family members observed by neowise


matchids = []
matchids_curlformat = []
nomatch = []

#for i in range(len(u_name_mpced_n)):
for i in range(len(objid_mpec)):

    imatch = np.where(u_name_mpced_n == objid_mpec[i])[0]
    if len(imatch) > 0:
        matchids.append(u_name_mpced_n[imatch][0])
        if u_objnum_n[imatch] == '0':
            matchids_curlformat.append(u_provdesig_n[imatch][0])
        elif u_objnum_n[imatch] != '0':
            matchids_curlformat.append(u_objnum_n[imatch][0])
        else:
            print('problem with ',u_objnum_n[imatch][0])

    else:
        nomatch.append(objid_mpec[i])



nomatch = np.asarray(nomatch)
matchids = np.asarray(matchids)
matchids_curlformat = np.asarray(matchids_curlformat)


# In[342]:


print(str(len(matchids))+' asteroids in this family found in the WISE/NEOWISE catalog')


# # This section is code to be executed for a single object and ultimately belongs in a for loop

# In[60]:


## determine the indices of the data columns

def determine_column_indices(colheads,cat):

    colnames = colheads.split('|')[1:]                    ## the [1:] is needed because the column headers contain a space in front of the text
    colnames = [colname.strip() for colname in colnames]

    imjd = colnames.index('mjd')
    icc_flags = colnames.index('cc_flags')
    iph_qual = colnames.index('ph_qual')

    if cat == 'neowiser_p1bs_psd':

        iwbflg = colnames.index('w1flg_1')
        iwbmpro = colnames.index('w1mpro')
        iwbsigmpro = colnames.index('w1sigmpro')
        iwbsnr = colnames.index('w1snr')

        iwrflg = colnames.index('w2flg_1')
        iwrmpro = colnames.index('w2mpro')
        iwrsigmpro = colnames.index('w2sigmpro')
        iwrsnr = colnames.index('w2snr')

    elif cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':

        iwbflg = colnames.index('w2flg_1')
        iwbmpro = colnames.index('w2mpro')
        iwbsigmpro = colnames.index('w2sigmpro')
        iwbsnr = colnames.index('w2snr')

        iwrflg = colnames.index('w3flg_1')
        iwrmpro = colnames.index('w3mpro')
        iwrsigmpro = colnames.index('w3sigmpro')
        iwrsnr = colnames.index('w3snr')

    else:
        print('Catalog name not recognized')

    return (imjd,icc_flags,iph_qual,iwbflg,iwbmpro,iwbsigmpro,iwbsnr,iwrflg,iwrmpro,iwrsigmpro,iwrsnr)


# In[61]:


def convert_to_letter(text):
    if int(text) <= 35:
        rtn = str(chr(int(text)+55))
    elif int(text) <= 75:
        rtn = str(chr(int(text)+61))
    else:
        print('conversion not found')

    return str(rtn)    


# In[62]:


def convert_to_mpecname(objid):

    newobjid = list(convert_to_letter(objid[0:2]))
    newobjid += objid[2:4]
    if len(objid) == 6:
        newobjid += objid[4]+'00'+objid[5]
    elif len(objid) == 7:
        newobjid += objid[4]+'0'+objid[6]+objid[5]
    elif len(objid) == 8:
        newobjid += objid[4]+objid[6:]+objid[5]
    elif len(objid) == 9:
        newobjid += objid[4]+convert_to_letter(objid[6:8])+objid[8]+objid[5]

    return "".join(newobjid)



# In[63]:


def get_positions(objid,jd_def):

    if len(objid) > 5:
        if objid[0:4].isdigit() and objid[4:6].isalpha():
            objid = convert_to_mpecname(objid)


    t = Time(jd_def, format='jd')
    utc = t.to_datetime()

    ast_positions = get_horizons_coord(objid, time = utc)
    astxyz = ast_positions.represent_as(CartesianRepresentation)

    wise_positions = get_horizons_coord('-163', time=utc)
    wisexyz = wise_positions.represent_as(CartesianRepresentation)

    sun_positions = get_horizons_coord('sun', time=utc)
    sunxyz = sun_positions.represent_as(CartesianRepresentation)

    a2s = sunxyz - astxyz
    a2o = wisexyz - astxyz

    ## Calculate the dot product of the unit vectors
    #dot_product = np.dot(a2s.xyz.to_value(u.au), a2o.xyz.to_value(u.au)) / \
    #              (np.linalg.norm(a2s.xyz.to_value(u.au)) * np.linalg.norm(a2o.xyz.to_value(u.au)))
    #
    ## Calculate the solar phase angle
    #phase_angle = np.arccos(dot_product) * u.rad
    #print(f"The solar phase angle for the asteroid is: {phase_angle.to(u.deg):.2f}")

    ##for j in range(len(wisex)):
    ##    print(astx[j],wisex[j],ast_to_wisex[j],asty[j],wisey[j],ast_to_wisey[j],astz[j],wisez[j],ast_to_wisez[j])

    return (a2s.x.value,
            a2s.y.value,
            a2s.z.value,
            a2o.x.value,
            a2o.y.value,
            a2o.z.value)



# In[76]:


def convert_mags_to_janskys(wbmag,wbmagerr,wrmag,wrmagerr,bflg):

    igood = np.where(bflg == 0)

    if len(igood) > 2:
        color = np.median(wbmag[igood]-wrmag[igood])
    else:
        color = np.median(wbmag - wrmag)

    if cat == 'neowiser_p1bs_psd':
        colorcode = np.array([-0.4040,-0.0538,0.2939,0.6393,0.9828,1.3246,1.6649,2.0041]) # color corrections from Wright et al. 2010 - Table 1, for W2    
        f_wb = [1.0283,1.0084,0.9961,0.9907,0.9921,1.,1.0142,1.0347]
        f_wr = [1.0206,1.0066,0.9976,0.9935,0.9943,1.,1.0107,1.0265]
    elif cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
        colorcode = np.array([-0.9624,-0.0748,0.8575,1.8357,2.8586,3.9225,5.0223,6.1524]) # color corrections from Wright et al. 2010 - Table 1, for W3    
        f_wb = [1.0206,1.0066,0.9976,0.9935,0.9943,1.,1.0107,1.0265]
        f_wr = [1.1344,1.0088,0.9393,0.9169,0.9373,1.,1.1081,1.2687]
    else:
        print('catalog name not recognized')

    diff = list(np.abs(colorcode-color))
    mn = min(diff)

    i_nu = diff.index(mn)
    print('i_nu = '+str(i_nu))

    if cat == 'neowiser_p1bs_psd':
        wbflux = (306.682/f_wb[i_nu])*10**(-wbmag/2.5)
        wrflux = (170.663/f_wr[i_nu])*10**(-wrmag/2.5)
    elif cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
        wbflux = (170.663/f_wb[i_nu])*10**(-wbmag/2.5)
        wrflux = (29.045/f_wr[i_nu])*10**(-wrmag/2.5)
    else:
        print('catalog name not recognized')

    wbfluxerr = wbflux * np.log(10) * wbmagerr
    wrfluxerr = wrflux * np.log(10) * wrmagerr

    return (wbflux,wbfluxerr,wrflux,wrfluxerr)



# In[77]:


def replace_null(arr,fill=-10.):

    try:
        arr = np.asarray(arr,dtype=float)
    except ValueError:

        arr_fix = []
        for val in arr:
            if val == 'null':
                arr_fix.append(fill)
            else:
                arr_fix.append(val)
        arr = np.asarray(arr_fix,dtype=float)

    return arr


# # Main cell to loop through the common objects in the family and neowise database

# In[343]:


for jj in range(int(istart),len(matchids)):

    print('Working on object ID '+str(matchids[jj])+', index = '+str(jj))

    ## request data from the IRSA website and store as a very long string - for a single object

    if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
        res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg')
    elif cat == 'neowiser_p1bs_psd':
        res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg')
    else:
        print('catalog not recognized')


    irsaoutput = res.text.splitlines()

    if len(irsaoutput) == 1:
        if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
            res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg')
        elif cat == 'neowiser_p1bs_psd':
            res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg')
        else:
            print('catalog not recognized')

        irsaoutput = res.text.splitlines()

        if len(irsaoutput) == 1:

            if len(matchids[jj]) > 5:
                if matchids[jj][0:4].isdigit() and matchids[jj][4:6].isalpha():
                    newname = convert_to_mpecname(matchids[jj])

            if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
                res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg')
            elif cat == 'neowiser_p1bs_psd':
                res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+':AST&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg')
            else:
                print('catalog not recognized')

            irsaoutput = res.text.splitlines()

            if len(irsaoutput) == 1:
                if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
                    res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg')
                elif cat == 'neowiser_p1bs_psd':
                    res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(matchids_curlformat[jj])+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg')
                else:
                    print('catalog not recognized')

                irsaoutput = res.text.splitlines()

    idata = []

    try:                

        icolhead = [i for i in range(len(irsaoutput)) if irsaoutput[i].startswith('|       cntr_u|')][0]
        colheads = irsaoutput[icolhead]

        for i in range(len(irsaoutput[icolhead:])):
            if irsaoutput[icolhead+i].startswith('|'):
                continue
            else:
                idata.append(i)

    except IndexError:

        print("Couldn't retrieve IRSA output for "+matchids[jj]+'.  Query returned: '+str(irsaoutput))


    if len(idata) > 0:

        idata = np.asarray(idata,dtype=int)[0]+icolhead
        datalines = irsaoutput[idata:]

        imjd,icc_flags,iph_qual,iwbflg,iwbmpro,iwbsigmpro,iwbsnr,iwrflg,iwrmpro,iwrsigmpro,iwrsnr = determine_column_indices(colheads,cat)

        mjd = np.asarray([float(line.split()[imjd]) for line in datalines])
        jd = mjd + 2400000.5
        cc_flags = np.asarray([str(line.split()[icc_flags]) for line in datalines])
        ph_qual = np.asarray([str(line.split()[iph_qual]) for line in datalines])
        wbflg_1 = np.asarray([line.split()[iwbflg] for line in datalines])
        wrflg_1 = np.asarray([line.split()[iwrflg] for line in datalines])

        wbmpro = np.asarray([line.split()[iwbmpro] for line in datalines])
        wbsigmpro = np.asarray([line.split()[iwbsigmpro] for line in datalines])
        wbsnr = np.asarray([line.split()[iwbsnr] for line in datalines])                                

        wrmpro = np.asarray([line.split()[iwrmpro] for line in datalines])
        wrsigmpro = np.asarray([line.split()[iwrsigmpro] for line in datalines])
        wrsnr = np.asarray([line.split()[iwrsnr] for line in datalines])

        flgs = np.asarray([wbflg_1,wrflg_1])        
        foo2 = np.empty((2,len(flgs[0,:])))
        foo2[0,:] = [int(flgs[0,iii].replace('null','9')) for iii in range(len(flgs[0,:]))]
        foo2[1,:] = [int(flgs[1,iii].replace('null','9')) for iii in range(len(flgs[0,:]))]
        flgs = np.asarray(foo2,dtype=int)


        jd_f = []
        wbmpro_f = []
        wrmpro_f = []
        wbsigmpro_f = []
        wrsigmpro_f = []
        bflgs_f = []

        for i in range(len(mjd)):

            if (cc_flags[i][ifilt] == '0') or (cc_flags[i][ifilt] == 'p') or (cc_flags[i][ifilt] == 'P'):
                if (ph_qual[i][ifilt] == 'A') or (ph_qual[i][ifilt] == 'B') or (ph_qual[i][ifilt] == 'C'):
                    if flgs[1,i] == 0:                            
                        jd_f.append(jd[i])
                        wbmpro_f.append(wbmpro[i])
                        wbsigmpro_f.append(wbsigmpro[i])
                        wrmpro_f.append(wrmpro[i])
                        wrsigmpro_f.append(wrsigmpro[i])
                        bflgs_f.append(flgs[0,i])
                    else:
                        continue

        jd_f = np.asarray(jd_f,dtype=float)

        wbmpro_f = replace_null(wbmpro_f)
        wbsigmpro_f = replace_null(wbsigmpro_f)
        wrmpro_f = replace_null(wrmpro_f)
        wrsigmpro_f = replace_null(wrsigmpro_f)

        bflgs_f = np.asarray(bflgs_f,dtype=int)

        if len(wbmpro_f) >= min_obs:

            #try:  wbmag,wrmag,wbmagerr,wrmagerr,bflg
            wbflux,wbfluxerr,wrflux,wrfluxerr = convert_mags_to_janskys(wbmpro_f,wbsigmpro_f,wrmpro_f,wrsigmpro_f,bflgs_f)
            #except:
            #
            #    for ii in range(len(filtereddata[:,1])):
            #        #print("%7.6f %1.8f %1.8f %1.8f %2.3f %1.3f" % (filtereddata[i,0],wbflux[i],wbfluxerr[i],wrflux[i],wrfluxerr[i],filtereddata[i,1],filtereddata[i,2]))
            #        print(str(filtereddata[ii,0]), str(wbflux[ii]), str(wbfluxerr[ii]), str(wrflux[ii]), str(wrfluxerr[ii]), str(filtereddata[ii,1]), str(filtereddata[ii,2]))
            #        print('error in trying to get the fluxes')
            #        sys.exit()

            try:
                astx,asty,astz,ast_to_wisex,ast_to_wisey,ast_to_wisez = get_positions(matchids_curlformat[jj],jd_f)

                wfile = open('Fam'+famid+'_data_'+cat+'_'+filterpriority+'/'+matchids_curlformat[jj]+'.obs','w+')
                wfile.write(str(int(len(wbflux)))+'\n')

                for ii in range(len(astx)):

                    wfile.write(str(jd_f[ii])+' 1'+'\n')
                    wfile.flush()

                    wfile.write("%1.8f %1.8f %1.8f\n" % (-astx[ii],-asty[ii],-astz[ii]))
                    wfile.write("%1.8f %1.8f %1.8f\n" % (-ast_to_wisex[ii],-ast_to_wisey[ii],-ast_to_wisez[ii]))
                    wfile.flush()
                    if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
                        if filterpriority == 'w2':
                            wfile.write('4.6028 '+str(round(wbflux[ii],10))+' '+str(round(wbfluxerr[ii],10))+' 1'+'\n')
                            wfile.flush()
                        if filterpriority == 'w3':
                            wfile.write('11.0984 '+str(round(wrflux[ii],10))+' '+str(round(wrfluxerr[ii],10))+' 2'+'\n')
                            wfile.flush()
                    elif cat == 'neowiser_p1bs_psd':
                        wfile.write('4.6028 '+str(round(wrflux[ii],10))+' '+str(round(wrfluxerr[ii],10))+' 1'+'\n')
                        wfile.flush()
                    else:
                        continue

                    wfile.write('\n')
                    wfile.write('\n')        

                wfile.close()

            except ValueError:
                print('No horizons match for obj id '+matchids[jj])

        else:

            print('Not enough filtered measurements for object ID '+matchids[jj]+' in this catalog: '+cat)

            wfile = open('Fam'+famid+'_data_'+cat+'_'+filterpriority+'/Nofilter_'+matchids[jj]+'.obs','w+')
            for jjj in range(len(irsaoutput)):
                wfile.write(irsaoutput[jjj])
                wfile.write('\n')
                wfile.flush()
            wfile.close()

    else:
        print('No data matches for '+matchids[jj]+' in this catalog: '+cat)




# # Auxiliary code from this point on

# In[96]:


jj = 621
print(len(matchids[jj]))
print(matchids[jj])

if len(matchids[jj]) > 5:
    print('hi')
    if matchids[jj][0:4].isdigit() and matchids[jj][4:6].isalpha():
        newname = convert_to_mpecname(matchids[jj])


# In[269]:


test = True

objid = '2005UK228'
objid = 'K02V21U'

if test:

    ## request data from the IRSA website and store as a very long string - for a single object

    if cat == 'allsky_4band_p1bs_psd' or cat == 'allsky_3band_p1bs_psd':
        res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(objid)+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg')
    elif cat == 'neowiser_p1bs_psd':
        res = requests.get('https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&mobj=smo&mobjstr='+str(objid)+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg')
    else:
        print('catalog not recognized')

    irsaoutput = res.text.splitlines()

    try:                

        icolhead = [i for i in range(len(irsaoutput)) if irsaoutput[i].startswith('|       cntr_u|')][0]
        colheads = irsaoutput[icolhead]

        for i in range(len(irsaoutput[icolhead:])):
            if irsaoutput[icolhead+i].startswith('|'):
                continue
            else:
                idata.append(i)

    except IndexError:

        print("Couldn't retrieve IRSA output for "+matchids[jj]+'.  Query returned: '+str(irsaoutput))


# In[262]:


print(len(irsaoutput))

if irsaoutput[1].startswith('['): 
    print('hi')


# In[249]:


for ij in range(len(irsaoutput)):
    print(str(ij)+' --> '+irsaoutput[ij])



# In[81]:


for ij in range(len(wbmpro_f)):
    print(str(ij),wbmpro_f[ij],wbsigmpro_f[ij],wrmpro_f[ij],wrsigmpro_f[ij],bflgs_f[ij])


# In[89]:


## useful code for querying the wise database for general frame information

from astroquery.ipac.irsa.most import Most
obs = Most.query_object(obj_name='818',
                        output_mode="Brief",
                        catalog='wise_allsky_4band',
                        obj_type='Asteroid',
                        )
print(obs.keys())


# In[10]:


## useful function for converting an MPC-ed letter code to a number

def convertMPCed(text):
    if text.isupper():
        rtn = ord(text)-55
    elif text.islower():
        rtn = ord(text)-61

    return str(rtn)


# In[ ]:


## write a curl script that would download the desired irsa tables

wfile = open(outfile,'w+')

if cat == 'allsky_4band_p1bs_psd':
    for i in range(len(matchids)):
        wfile.write('curl -o '+str(matchids[i])+'.tbl "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&moradius=0.3&mobj=smo&mobjstr='+str(matchids[i])+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w4flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,w3mpro,w3sigmpro,w3snr,w3flg,w4mpro,w4sigmpro,w4snr,w4flg'+'"\n')
        wfile.write("\n")
        wfile.flush()
elif cat == 'neowiser_p1bs_psd':
    for i in range(len(matchids)):
        wfile.write('curl -o '+str(matchids[i])+'.tbl "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&searchForm=MO&spatial=cone&catalog='+cat+'&moradius=0.3&mobj=smo&mobjstr='+str(matchids[i])+'&selcols=mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w4flg_1,w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg'+'"\n')
        wfile.write("\n")
        wfile.flush()
else: 
    print('Catalog defined is not recognized')

wfile.close()

