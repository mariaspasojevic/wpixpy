import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

def read_orbit( orbitfile ):
  '''
  df = read_orbit( orbitfile ):

  Inputs:
    orbitfile = full path to file of format like:
                'DSX_17213000000_17213235959_EPH_00_L1.txt'
  Outputs:
    Tuple containing (status, df)
    status = 1 if valid orbit dataframe returned  
    orbit = dataframe containing all variables from the emphemeris file
  '''

  # Constants
  R_E = 6371    # Earth Radius in km

  # check if file found
  if not os.path.isfile( orbitfile ):
    print('%s: inputfile %s not found' \
           % (inspect.currentframe().f_code.co_name, hdf_L0_filename) )
    return (-1,0)
  else:
    f = open(orbitfile, 'r')

  # Read 13 lines of header
  for i in range(0,12):
    f.readline()

  # 4-line record format

  # Line 1: orbital ephemeris
  # Read variable names
  ephem = f.readline()
  ephem = ephem[2:-1].split()
  f.readline()

  # Line 2: space environment
  # Read variable names
  space_env = f.readline()
  space_env = space_env[2:-1].split()
  f.readline()

  # Line 3: conjunctions
  # Read variable names
  conjunct = f.readline()
  conjunct = conjunct[2:-1].split()
  f.readline()

  # Line 4: blank
  f.readline()

  # Read records combining the 3 lines skipping 4th.
  records = [f.readline().split() + f.readline().split() + f.readline().split()]
  f.readline()
  nextline = f.readline()
  while nextline:
    records.append(nextline.split()+f.readline().split() + f.readline().split())
    f.readline()
    nextline = f.readline()

  f.close()

  # Convert list of lists into dataframe using column names
  orbit = pd.DataFrame(records, columns = ephem+space_env+conjunct)

  # Combine two datetime columns, drop star column
  orbit['YYYYDDD'] = orbit.apply(lambda x: \
    datetime.datetime.strptime(x['YYYYDDD']+x['HHMMSS.UUU'], '%Y%j%H%M%S.%f'), axis=1)
  orbit.rename(columns={'YYYYDDD': 'UT'}, inplace=True)
  orbit.drop(['*','HHMMSS.UUU'], axis=1, inplace=True)

  # remove *, plus two date columns from ephem
  ephem = ephem[3::]
  
  # convert datatype of all columns
  # separate variables by type
  ephem_flags = [x for x in ephem if 'FLAG' in x]
  ephem_num = [x for x in ephem if 'NUM' in x]
  ephem_dce = ['PLAN_DCE']
  ephem = [x for x in ephem if x not in ephem_flags+ephem_num+ephem_dce]
  
  space_env_flags = [x for x in space_env if 'FLAG' in x]
  space_env = [x for x in space_env if x not in space_env_flags]
  
  # convert to correct types
  orbit[ephem+space_env] = orbit[ephem+space_env].astype(float)
  orbit[ephem_flags+space_env_flags+conjunct] = \
        orbit[ephem_flags+space_env_flags+conjunct].astype(int)
  orbit[ephem_num] = orbit[ephem_num].astype(int)
  orbit[ephem_dce] = orbit[ephem_dce].astype(str)

  # Convert SM X, Y, Z from km to R_E
  sm = ['SM_X','SM_Y', 'SM_Z']
  orbit.loc[:,sm] = orbit[sm].apply( lambda x: x/R_E )

  return (1,orbit)

# exec(open('dsx.py').read())

basepath = '/Users/mystical/Work/DSX/Data/Ephemeris/'
orbitfile = basepath + 'DSX_17213000000_17213235959_EPH_00_L1.txt'
df = read_orbit(orbitfile)[1]

plt.subplot(221)
plt.plot(df.SM_X, df.SM_Y)
plt.axis('equal'), plt.axis('square')
plt.xlabel('X, R_E')
plt.ylabel('Y, R_E')

plt.subplot(223)
plt.plot(df.SM_X, df.SM_Z)
plt.axis('equal'), plt.axis('square')
plt.xlabel('X, R_E')
plt.ylabel('Z, R_E')

plt.subplot(224)
plt.plot(df.SM_Y, df.SM_Z)
plt.axis('equal'), plt.axis('square')
plt.xlabel('Y, R_E')
plt.ylabel('Z, R_E')

plt.show()
    





