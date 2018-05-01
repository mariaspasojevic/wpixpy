import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def dsx_read_orbit( orbitfile ):
	'''
	df = dsx_read_orbit( orbitfile ):

	Inputs:
		orbitfile = full path to file of format like:
								'DSX_17213000000_17213235959_EPH_00_L1.txt'
	Outputs:
		df = dataframe containing all variables from the emphemeris file
	'''

	f = open(orbitfile,'r')

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
	df = pd.DataFrame(records, columns = ephem+space_env+conjunct)

	# Combine two datetime columns, drop star column
	df['YYYYDDD'] = df.apply(lambda x: \
		datetime.datetime.strptime(x['YYYYDDD']+x['HHMMSS.UUU'], '%Y%j%H%M%S.%f'), axis=1)
	df.rename(columns={'YYYYDDD': 'UT'}, inplace=True)
	df.drop(['*','HHMMSS.UUU'], axis=1, inplace=True)

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
	df[ephem+space_env] = df[ephem+space_env].astype(float)
	df[ephem_flags+space_env_flags+conjunct] = \
				df[ephem_flags+space_env_flags+conjunct].astype(int)
	df[ephem_num] = df[ephem_num].astype(int)
	df[ephem_dce] = df[ephem_dce].astype(str)

	return df

# exec(open('dsx_orbit.py').read())
orbitfile ='DSX_17213000000_17213235959_EPH_00_L1.txt'
df = dsx_read_orbit(orbitfile)

plt.subplot(131)
plt.plot(df.SM_X, df.SM_Y)
plt.axis('equal')

plt.subplot(132)
plt.plot(df.SM_X, df.SM_Z)
plt.axis('equal')

plt.subplot(133)
plt.plot(df.SM_Y, df.SM_Z)
plt.axis('equal')

plt.show()
		





