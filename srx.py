#!/usr/bin/env python
"""
srx.py

Common functions used for working with SRX data

"""

import os
import numpy as np
import pandas as pd
import datetime
import h5py
import inspect
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

######

def read_hdf_L0(hdf_L0_filename=[],           \
                group_name='SVY',             \
                dataset_name='svyUnpacked1',  \
                attribute_name='svyHdr'):
  """
  (status, srx) = srx_read_hdf_L0( hdf_L0_filename, 
                                   group_name, 
                                   dataset_name, 
                                   attribute_name)
  Reads L0 SRX file

  Inputs:
    hdf_L0_filename = name of unpacked L0 hdf file
         group_name = e.g., 'SVY', 'MBA'
       dataset_name = e.g., 'svyUnpacked1'
     attribute_name = e.g., 'svyHdr'

  Outputs:
    Tuple with (status, srx)
    status     =  1, valid srx dictionary returned
               = -1, file not found, no file selected
               = -2, invalid format
    srx: dictionary with keys depending on data products
  """ 

  pyname = inspect.currentframe().f_code.co_name
    
  # if filename blank, use a gui to select
  if not hdf_L0_filename:
    root = tk.Tk()
    hdf_L0_filename = filedialog.askopenfilename(initialdir = os.getcwd() )
    root.destroy()
                
    # exit if no file selected
    if not hdf_L0_filename:
      print('%s: cancelled (no file selected)' %pyname)
      return (-1, 0)
    
  # check if file found
  if not os.path.isfile( hdf_L0_filename ):
    print('%s: inputfile %s not found' % (pyname, hdf_L0_filename) )
    return (-1, 0)
        
  # check file type
  f_path, f_name = os.path.split(hdf_L0_filename)
  f_basename, f_ext = os.path.splitext(f_name)

  IS_HDF = (f_ext == '.h5') | (f_ext == '.hdf')

  if not IS_HDF:
    print('%s: inputfile %s not HDF' % (pyname, hdf_L0_filename) )
    return (-2, 0)

  # open hdf
  data = h5py.File(hdf_L0_filename, 'r')

  # check for Groups
  if list(data.keys())[0] != group_name:
    print('%s: No %s  Groups found %s' % 
      (inspect.currentframe().f_code.co_name, group_name, hdf_L0_filename))
    data.close()
    return (-2, 0)

  # get dataset and header
  dataset = data[group_name][dataset_name]
  header = dataset.attrs[attribute_name]

  # convert to dictionary
  srxHdr = [dict(zip(header.dtype.names,x)) for x  in header][0]

  # SVY
  if group_name == 'SVY':
    output = read_svy_blks( srxHdr, dataset )
  # MBA
  elif group_name == 'MBA':
    output = read_mba_blks( srxHdr, dataset )

  # check if valid output was returned
  if output[0]:
    srxBlk = output[1]
  else:
    data.close()
    return (-2, 0)

  # close hdf file
  data.close()

  # combine both dictionaries
  srx = {**srxHdr, **srxBlk}

  # add filename
  srx[group_name.lower() + 'File'] = hdf_L0_filename

  return (1,srx)

########

def read_svy_blks( svyHdr, svyBlk ):

  # constants
  SRX_SAMPLE_RATE_C = 100e3  # BBR sample rate Hz

  # Convert to dictionary
  svy = [dict(zip(svyBlk.dtype.names,x)) for x in svyBlk][0]
  
  ## Add additional columns to svy dict
  # tranpose svyImage, freq: columns, time: rows
  svy['svyImage'] = svy['svyImage'].T

  # n_rows = H.svyVpixels, n_cols = H.svyHpixels IFF full image
  n_rows, n_cols = svy['svyImage'].shape

  # interpret unixtime
  svy['time_str'] = datetime.datetime.utcfromtimestamp( \
      svyHdr['svyUnixtime'] ).strftime('%Y-%m-%d %H:%M:%S')

  # freq res (Vpixels=fftSize/2)
  svy['df'] = (SRX_SAMPLE_RATE_C/2)/svyHdr['svyVpixels']

  # frequency-axis
  svy['F'] = svy['df'] * np.arange( 0, svyHdr['svyVpixels'] )

  # time-axis (per #lines read)
  svy['T'] = svyHdr['svyInterval']*np.arange(0, n_cols)

  # image array (as double)
  svy['S'] = svy['svyImage'].astype(float)

  # channel name
  chName = ['Bx','By','Bz','Ey','Ez']
  svy['svyChanName'] = chName[ svyHdr['svyChanNum'] ]

  return (1, svy)

########

def read_mba_blks( mbaHdr, mbaBlk ):

  # derive a few parameters needed to plot S-matrix
  mbaCh  = mbaHdr['mbaChanSel']          # short name for convenience
  nBands = mbaHdr['mbaNedges'] - 1       # to confirm mbaBlk[*].nBands
  nBlks  = mbaBlk.shape[0]               # number of sample-intervals

  # determine Stype and nElem per mbaBlk[b] from mbaChanSel
  # Stype conveys which Re{Sij},Im{Sij} calculated vs omitted
  if (mbaCh == 95) or (mbaCh == 127):    # bit6=On with all 5ch
    Stype = 3        # Stype 3 = sparse, some Sij omitted
    nElem = 12       # 12 of 15 elements
  elif (mbaCh == 63):
    Stype = 2        # Stype 2 = reduced, some Re,Im = zero
    nElem = 15       # 15 elements
  elif (mbaCh == 31):
    Stype = 1        # Stype 1 = full, all Re,Im calculated
    nElem = 15       # 15 elements
  else:
    Stype = 0        # Stype 0 = none (main-diag only)
    nElem = bin(mbaCh).count("1")  # count Ch = On (<5)
  
  # init 4d Sij(freq, time, i, j) matrix, use triu if mem tight
  S_ftij = np.zeros((nBands,nBlks,5,5), dtype=complex) # init S-matrix

  # sift mbaBlk[*].S_ij to Sij per enabled channels
  for b in range(0, nBlks):                 # loop mbaBlk[*]
    for n in range(0, nElem):             # loop Sij elements
      mba_ch = mbaBlk[b]['mba_ch']      # grab this mba_ch

      if mba_ch['nBands'][0] != nBands: # consistency check
        return (-1, 0)

      ch_ij = mba_ch['ch_ij'][n]        # map n^th element

      j = ch_ij % 10 - 1                # ones digit always
      if ch_ij < 10:                    # tens digit depends
        i = j                         # i index = copy j
      else:
        i = int(ch_ij/10) - 1         # i index explicit

      S_ij = mba_ch['S_ij'][n]          # grab n^th S_ij data

      x = S_ij['real']                  # Re{S_ij} part
      y = S_ij['imag']                  # Im{S_ij} part

      S_ftij[:,b,i,j] = x + 1j * y      # complex assign

  # time-axis simple multiple of mbaInterval
  T = mbaHdr['mbaInterval']*np.arange(0, nBlks)/60 # time-axis (min)

  # freq-axis = geometric mean of breakpoints
  F = np.arange(0, nBands, dtype=np.float)         # freq-axis
  df = 100e3 / mbaHdr['mbaFFTsize']                # basic FFT resolution
  Fedge = df * mbaHdr['mbaEdgeIdx']                # MBA log-space breakpoints
  for k in range(0, nBands):
    if Fedge[k] == 0:
        F[k] = Fedge[k+1]/2                      # midpoint
    else:
        F[k] = np.sqrt(Fedge[k] * Fedge[k+1])    # geometric mean


  mba = {'F':F, 'T':T, 'S_ftij':S_ftij, 'Stype':Stype}

  # interpret unixtime
  mba['time_str'] = datetime.datetime.utcfromtimestamp( \
      mbaHdr['mbaUnixtime'] ).strftime('%Y-%m-%d %H:%M:%S')

  return (1,mba)

  
