# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:39:52 2015

@author: Surfinfan
"""

import pandas
import numpy

# Read the data from the csv file
data = pandas.read_csv('../DataSet/nesarc_pds.csv', low_memory=False)

# Length of the dataset and number of columns
print (len(data))
print (len(data.columns))

# Let's be sure all the info is numerical or NA
data['S1Q1E'] = data['S1Q1E'].convert_objects(convert_numeric = True)
data['S2AQ4A'] = data['S2AQ4A'].convert_objects(convert_numeric = True)
data['S2AQ4B'] = data['S2AQ4B'].convert_objects(convert_numeric = True)
data['S2AQ5A'] = data['S2AQ5A'].convert_objects(convert_numeric = True)
data['S2AQ5B'] = data['S2AQ5B'].convert_objects(convert_numeric = True)
data['S2AQ6A'] = data['S2AQ6A'].convert_objects(convert_numeric = True)
data['S2AQ6B'] = data['S2AQ6B'].convert_objects(convert_numeric = True)
data['S2AQ7A'] = data['S2AQ7A'].convert_objects(convert_numeric = True)
data['S2AQ7B'] = data['S2AQ7B'].convert_objects(convert_numeric = True)


# Now there are blocks of code, basically all very similar, in order to
# show distribution of different variables of interest for me
# First in absolute value, then in %

#----------------------------COUNTRIES OF ORIGIN-------------------------------
c_paises = data['S1Q1E'].value_counts(sort=False)
print ("Distribution of country of ORIGIN OR DESCENT")
print (c_paises)
p_paises = data['S1Q1E'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of country of ORIGIN OR DESCENT [%]")
print (p_paises)


# ----------------------------COOLERS------------------------------------------
c_coolers = data['S2AQ4A'].value_counts(sort=False, dropna = False)
print ("Distribution of DRANK ANY COOLERS IN LAST 12 MONTHS")
print (c_coolers)
p_coolers = data['S2AQ4A'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of DRANK ANY COOLERS IN LAST 12 MONTHS [%]")
print (p_coolers)

c_coolers_how = data['S2AQ4B'].value_counts(sort=False, dropna = False)
print ("Distribution of HOW OFTEN DRANK COOLERS IN LAST 12 MONTHS")
print (c_coolers_how)
p_coolers_how = data['S2AQ4B'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of HOW OFTEN DRANK COOLERS IN LAST 12 MONTHS [%]")
print (p_coolers_how)


# ----------------------------BEERS------------------------------------------
c_beers = data['S2AQ5A'].value_counts(sort=False, dropna = False)
print ("Distribution of DRANK ANY BEER IN LAST 12 MONTHS")
print (c_beers)
p_beers = data['S2AQ5A'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of DRANK ANY BEER IN LAST 12 MONTHS [%]")
print (p_beers)

c_beers_how = data['S2AQ5B'].value_counts(sort=False, dropna = False)
print ("Distribution of HOW OFTEN DRANK BEERS IN LAST 12 MONTHS")
print (c_beers_how)
p_beers_how = data['S2AQ5B'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of HOW OFTEN DRANK BEERS IN LAST 12 MONTHS [%]")
print (p_beers_how)


# ----------------------------WINE------------------------------------------
c_wine = data['S2AQ6A'].value_counts(sort=False, dropna = False)
print ("Distribution of DRANK ANY WINE IN LAST 12 MONTHS")
print (c_wine)
p_wine = data['S2AQ6A'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of DRANK ANY WINE IN LAST 12 MONTHS [%]")
print (p_wine)

c_wine_how = data['S2AQ6B'].value_counts(sort=False, dropna = False)
print ("Distribution of HOW OFTEN DRANK WINE IN LAST 12 MONTHS")
print (c_wine_how)
p_wine_how = data['S2AQ6B'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of HOW OFTEN DRANK WINE IN LAST 12 MONTHS [%]")
print (p_wine_how)


# ----------------------------LIQUOR------------------------------------------
c_liquor = data['S2AQ7A'].value_counts(sort=False, dropna = False)
print ("Distribution of DRANK ANY LIQUOR IN LAST 12 MONTHS")
print (c_liquor)
p_liquor = data['S2AQ7A'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of DRANK ANY LIQUOR IN LAST 12 MONTHS [%]")
print (p_liquor)

c_liquor_how = data['S2AQ7B'].value_counts(sort=False, dropna = False)
print ("Distribution of HOW OFTEN DRANK LIQUOR IN LAST 12 MONTHS")
print (c_liquor_how)
p_liquor_how = data['S2AQ7B'].value_counts(sort=False, normalize = True, dropna = False)
print ("Distribution of HOW OFTEN DRANK LIQUOR IN LAST 12 MONTHS [%]")
print (p_liquor_how)

