#!/usr/bin/env python
"""
mba.py

Functions used for working with MBA data

"""

import srx
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import h5py
import inspect

###
# quicklook_plot
###
def quicklook_plot( inputfile=[], dset_num = 1, pngpath='' ):

  """
  (status, mba) = quicklook_plot( inputfile=[], dset_num=1, pngpath='' ):

  Inputs:
    inputfile = unpacked MBA L0 file
     dset_num = dataset number only if not 1 (.h5 only)
      pngpath = path to save plot

  Outputs:
    Tuple containing:
    status =  1 valid mba dictionary is returned
           = -1, file not found, no file selected
           = -2, invalid format
       mba = dictionary as returned by srx.read_hdf_L0() with 1 additional key
             'pngname' is name of file containing plot of data

  """
  # Get name of current function
  pyname = inspect.currentframe().f_code.co_name

  # Set Seaborn parameters
  sns.set()
  sns.set(font_scale=0.8)

  # hdf structure
  group_name = 'MBA'
  dataset_name = 'mbaUnpacked%d' % dset_num
  attribute_name = 'mbaFileHdr'

  # read hdf file
  output = srx.read_hdf_L0( inputfile, group_name, \
                            dataset_name, attribute_name)

  # Check if valid svy dictionary returned
  if output[0] == 1:
    mba = output[1]
  else:
    return output

  # split filename into parts
  f_path, f_name = os.path.split(mba['mbaFile'])
  f_base, f_ext = os.path.splitext(f_name)

  myeps = 1e-12                             # avoid log10(zero) below
  yMin  = 100                               # 100Hz practical lower f
  yMax  = 50e3                              # 50Khz maximum upper f
  dBmin = 0                                 # ax.pcolormesh min dB
  dBmax = 100                               # ax.pcolormesh max dB

  ## PLOT
  plt.close('all')
  fig = plt.figure(1, figsize=(11, 8.5))

  for i in range(0, 5):
    for j in range(i, 5):

      S = mba['S_ftij'][:,:,i,j]              # this Sij 2-d

      # Plot Main Diagonal
      if i == j:                       
        ax = plt.subplot(5,5,i+1+5*j)  

        SdB = 10 * np.log10(S.real + myeps) # 10*log10() since |X|^2 term
        ax.pcolormesh(mba['T'], mba['F'], SdB, cmap='jet', \
                      vmin=dBmin, vmax=dBmax)
        ax.set_yscale('log')
        ax.set_ylim([yMin, yMax])
        ax.set_title("S({0},{1})".format(i+1, j+1), fontsize=10 )
        
        ax.set_xlabel('Time, min')
        ax.set_ylabel('Freq, Hz')

      else:
        if mba['Stype'] < 1:  # skip if no off-diags
          continue

        # Plot Real and Imag part of off-diagonals
        for k in ['Imag', 'Real']:
          if k == 'Imag':
            ax = plt.subplot(10, 5, j+1+10*i)
            SdB = 10 * np.log10(np.abs(S.imag) + myeps) # 10*log10 since XiXj
          else:
            ax = plt.subplot(10, 5, j+1+10*i+5)
            SdB = 10 * np.log10(np.abs(S.real) + myeps) # 10*log10 since XiXj
  
          ax.pcolormesh(mba['T'], mba['F'], SdB, \
                        cmap='jet', vmin=dBmin, vmax=dBmax)

          ax.set_yscale('log')
          ax.set_ylim([yMin, yMax])
          ax.set_title(k + ' S({0},{1})'.format(i+1, j+1), fontsize=10)
          ax.axes.get_xaxis().set_ticks([])
          ax.axes.get_yaxis().set_ticks([])


  now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  title_str = 'MBA File: %s, date: %s, QL run: %s' % \
    (f_name, mba['time_str'], now_str)
  plt.suptitle(title_str)

  plt.show(block=False)
  
  ## save png file
  if not pngpath:
    pngpath = f_path

  mba['pngname'] = pngpath + f_base + "_S{0}{1}".format(i+1, j+1) +  ".png"
  plt.savefig(mba['pngname'], format='png')

  return (1, mba)


# exec(open('mba.py').read())

files = ['20170130_232642_mba_1485817923.h5']      # 2ch, no S-matrix
files.append('20170130_232742_mba_1485817967.h5')  # 5ch, full S-matrix
files.append( '20170130_232853_mba_1485818011.h5') # 5ch, reduced S-matrix
files.append( '20170130_233054_mba_1485818055.h5') # 5ch, sparse S-matrix
files.append( '20170720_220616_mba_1500587061.h5') # 2ch, live data, 1.5min

pathname = '/Users/mystical/Work/DSX/Software/srx_quicklook/mbaQuicklook_20180426/'

for f in files:
  output = quicklook_plot(pathname+f, pngpath=pathname)

