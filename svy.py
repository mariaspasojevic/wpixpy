#!/usr/bin/env python
"""
svy.py

Functions used for working with SVY data

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
  (status, svy) = quicklook_plot( inputfile=[], dset_num=1, pngpath='' ):

  Inputs:
    inputfile = unpacked SVY e.g. 'svy_1234567890.h5'
     dset_num = dataset number only if not 1 (.h5 only)
      pngpath = path to save plot

  Outputs:
    Tuple containing:
    status =  1 valid svy dictionary is returned 
           = -1, file not found, no file selected
           = -2, invalid format
    svy = dictionary as returned by srx.read_hdf_L0() with 1 additional key
          'pngname' is name of file containing plot of data

  """
  pyname = inspect.currentframe().f_code.co_name

  sns.set()
  sns.set(context='notebook', font_scale=1.2)

  # hdf structure
  group_name = 'SVY'
  dataset_name = 'svyUnpacked%d' % dset_num
  attribute_name = 'svyHdr'

  # Read HDF file
  output = srx.read_hdf_L0( inputfile, group_name, 
                            dataset_name, attribute_name)

  # Check if valid svy dictionary returned
  if output[0] == 1:
    svy = output[1]
  else:
    return output

  # split filename into parts
  f_path, f_name = os.path.split(svy['svyFile'])
  f_basename, f_ext = os.path.splitext(f_name)

  ## plot
  plt.close('all')
  fig, ax = plt.subplots(figsize=(11, 8.5))

  cax = ax.pcolormesh(svy['T']/60/60,svy['F']/1000,svy['S'], \
                      cmap='jet', vmin=0, vmax=255)

  now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  title_str = 'SVY %s \n  File: %s,   Ch %d of [0..4],   QL run %s' % \
    (svy['svyChanName'], f_name, svy['svyChanNum'], now_str)

  ax.set_title(title_str)
  ax.set_ylabel('Freq, kHz')
  ax.set_xlabel( 'Time, hrs from %s UT (%d sec cadence)' \
                % (svy['time_str'], svy['svyInterval'] ) )

  fig.colorbar(cax, label='Raw Units')

  fig.tight_layout()
  plt.show(block=False)     
  
  # save png file
  if not pngpath:
    pngpath = f_path + '/'

  svy['pngname'] = pngpath + f_basename + '.png'
  plt.savefig(svy['pngname'], format='png')

  return (1, svy)

# exec(open('svy.py').read())
pngpath = '/Users/mystical/Work/DSX/Software/TmpPlots/'
pathname = '/Users/mystical/Work/DSX/Software/' + \
            'srx_quicklook/svyQuicklook_20171211/'
files = ['20080901_135959_svy_1_1220277599.h5', \
         '20111117_153800_svy_1_1321544280.h5', \
         '20170323_013144_svy_1_1490230506.h5', \
         '20170323_013333_svy_3_1490230506.h5']

#for f in files:
#  output = quicklook_plot(inputfile=pathname+f, pngpath=pngpath)

output = quicklook_plot(inputfile=pathname+files[0], pngpath=pngpath)


