# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:39:52 2015

@author: Surfinfan

This is my work for the 'Data Management and Visualization' Coursera course
The question I want to study: how cultural origins influence in the way people 
chooses and uses alcoholic drinks. 

My hipothesis is that Yes, theres is a influence and different country-of-origin 
individuales will consume alcohol in a different way.

I will use the NESARC dataset considering following variables:

IDNUM  - UNIQUE ID NUMBER WITH NO ALPHABETICS
S1Q1E  - ORIGIN OR DESCENT
S2AQ4A - DRANK ANY COOLERS IN LAST 12 MONTHS
S2AQ4B - HOW OFTEN DRANK COOLERS IN LAST 12 MONTHS
S2AQ5A - DRANK ANY BEER IN LAST 12 MONTHS 
S2AQ5B - HOW OFTEN DRANK BEER IN LAST 12 MONTHS 
S2AQ6A - DRANK ANY WINE IN LAST 12 MONTHS
S2AQ6B - HOW OFTEN DRANK WINE IN LAST 12 MONTHS
S2AQ7A - DRANK ANY LIQUOR IN LAST 12 MONTHS
S2AQ7B - HOW OFTEN DRANK LIQUOR IN LAST 12 MONTHS

"""

import pandas
import numpy

# Read the data from the csv file
data = pandas.read_csv('../DataSet/nesarc_pds.csv', low_memory=False)


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

# Chicano and Mexican-American is the same thing actually
data['S1Q1E'] = data['S1Q1E'].replace(9, 36)      


# Let's create a new category [0] for people that don't drink at all
data['S2AQ4A'] = data['S2AQ4A'].replace(numpy.nan, 0)
data['S2AQ5A'] = data['S2AQ5A'].replace(numpy.nan, 0)
data['S2AQ6A'] = data['S2AQ6A'].replace(numpy.nan, 0)
data['S2AQ7A'] = data['S2AQ7A'].replace(numpy.nan, 0)


# Now, let's have in consideration that most of NaNs in 'B' series
# ['B' series of columns are the 'How often drank XXX ...'] are people that
# it is included in the group "I don't drink". The Codebook explains that for 
# 'B' series NaN mean "did not drink or unknown if drank XXX in last 12 months" 
# This includes both sub-groups, people that don't drink and people that does  
# not know if they have drinked XXX in the last 12 months. The first sub group 
# (abstainers) are identified in the 'A' series of columns with an 0, so, 
# let's separate both sub-groups in the 'B' series. 

# Functions to determine if the row is a real abstainer or non drinker 
# It returns 0 if the individual is already considered a non-drinker

def abstainer_coolers (row):
    if (row['S2AQ4A'] == 0):
        return 0
    else:
        return row['S2AQ4B']
        
def abstainer_beers (row):
    if (row['S2AQ5A'] == 0):
        return 0
    else:
        return row['S2AQ5B']

def abstainer_wine (row):
    if (row['S2AQ6A'] == 0):
        return 0
    else:
        return row['S2AQ6B']

def abstainer_liquor (row):
    if (row['S2AQ7A'] == 0):
        return 0
    else:
        return row['S2AQ7B']

# and now we get for each 'B' serie the individuals considered as abstainer
# marked with a 0 and not a NaN
data['S2AQ4B'] = data.apply(lambda row:abstainer_coolers(row), axis = 1)
data['S2AQ5B'] = data.apply(lambda row:abstainer_beers(row), axis = 1)
data['S2AQ6B'] = data.apply(lambda row:abstainer_wine(row), axis = 1)
data['S2AQ7B'] = data.apply(lambda row:abstainer_liquor(row), axis = 1)


# Let's put all data with no information into NaN too
data['S1Q1E'] = data['S1Q1E'].replace(98, numpy.nan)      #other
data['S1Q1E'] = data['S1Q1E'].replace(99, numpy.nan)      # unknown
data['S2AQ4A'] = data['S2AQ4A'].replace(9, numpy.nan)    # unknown
data['S2AQ4B'] = data['S2AQ4B'].replace(99, numpy.nan)    # unknown
data['S2AQ5A'] = data['S2AQ5A'].replace(9, numpy.nan)    # unknown
data['S2AQ5B'] = data['S2AQ5B'].replace(99, numpy.nan)    # unknown
data['S2AQ6A'] = data['S2AQ6A'].replace(9, numpy.nan)    # unknown
data['S2AQ6B'] = data['S2AQ6B'].replace(99, numpy.nan)    # unknown
data['S2AQ7A'] = data['S2AQ7A'].replace(9, numpy.nan)    # unknown
data['S2AQ7B'] = data['S2AQ7B'].replace(99, numpy.nan)    # unknown


# In the 'A' series of data, it seems that it would be more logical to have 
# a different order of dummy codes. Now, '0' means 'Don'k drik at all', 
# '1' means 'Yes, I Drunk XXX in the last 12 months' and '0' means 'No, I'm 
# not abstainer but I did not drink XXX in thelast 12 months'
# I would prefer a distribution like:
# '0' - Abstainer
# '1' - Not abstainer, but didn't drink XXX in the last 12 months
# '2' - Yes, drunk XXX in the last 12 months
# NaN - Don't know

recodeA = {0:0, 1:2, 2:1}
data['S2AQ4A'] = data['S2AQ4A'].map(recodeA)
data['S2AQ5A'] = data['S2AQ5A'].map(recodeA)
data['S2AQ6A'] = data['S2AQ6A'].map(recodeA)
data['S2AQ7A'] = data['S2AQ7A'].map(recodeA)

# In the 'B' series, the shorter the number in the dummy code, the higher the
# intensity of drinking. It would be more logical to invert the dummy codes:
# '10' - Every day
# '9'  - Nearly every day
# '8'  - 3 to 4 times a week
# '7'  -  2 times a week
# '6'  - Once a week
# '5'  - 2 to 3 times a month
# '4'  - Once a month
# '3'  - 7 to 11 times in the last year
# '2'  - 3 to 6 times in the last year
# '1'  - 1 to 2 times in the last year
# '0'  - Abstainer


recodeB = {0:0, 1:10, 2:9, 3:8, 4:7, 5:6, 6:5, 7:4, 8:3, 9:2, 10:1}
data['S2AQ4B'] = data['S2AQ4B'].map(recodeB)
data['S2AQ5B'] = data['S2AQ5B'].map(recodeB)
data['S2AQ6B'] = data['S2AQ6B'].map(recodeB)
data['S2AQ7B'] = data['S2AQ7B'].map(recodeB)


# Now there are blocks of code, basically all very similar, in order to
# show distribution of different variables of interest for me
# First in absolute value, then in %, then cumulative, then % cumulative

#----------------------------COUNTRIES OF ORIGIN-------------------------------
countries = pandas.DataFrame(data = data['S1Q1E'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
countries['percent'] = data['S1Q1E'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
countries['cumulative_frequency'] = countries['frequency'].cumsum()
countries['cumulative_percent'] = countries['percent'].cumsum()


# ----------------------------COOLERS------------------------------------------
coolers = pandas.DataFrame(data = data['S2AQ4A'].value_counts(sort=False, 
                           dropna = False), columns = ['frequency'])
coolers['percent'] = data['S2AQ4A'].value_counts(sort=False, 
                           normalize = True, dropna = False)*100
coolers['cumulative_frequency'] = coolers['frequency'].cumsum()
coolers['cumulative_percent'] = coolers['percent'].cumsum()

coolers_how = pandas.DataFrame(data = data['S2AQ4B'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
coolers_how['percent'] = data['S2AQ4B'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
coolers_how['cumulative_frequency'] = coolers_how['frequency'].cumsum()
coolers_how['cumulative_percent'] = coolers_how['percent'].cumsum()


# ----------------------------BEERS------------------------------------------
beers = pandas.DataFrame(data = data['S2AQ5A'].value_counts(sort=False, 
                         dropna = False), columns = ['frequency'])
beers['percent'] = data['S2AQ5A'].value_counts(sort=False, 
                         normalize = True, dropna = False)*100
beers['cumulative_frequency'] = beers['frequency'].cumsum()
beers['cumulative_percent'] = beers['percent'].cumsum()

beers_how = pandas.DataFrame(data = data['S2AQ5B'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
beers_how['percent'] = data['S2AQ5B'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
beers_how['cumulative_frequency'] = beers_how['frequency'].cumsum()
beers_how['cumulative_percent'] = beers_how['percent'].cumsum()


# ----------------------------WINE------------------------------------------
wine = pandas.DataFrame(data = data['S2AQ6A'].value_counts(sort=False, 
                        dropna = False), columns = ['frequency'])
wine['percent'] = data['S2AQ6A'].value_counts(sort=False, 
                        normalize = True, dropna = False)*100
wine['cumulative_frequency'] = liquor['frequency'].cumsum()
wine['cumulative_percent'] = liquor['percent'].cumsum()

wine_how = pandas.DataFrame(data = data['S2AQ6B'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
wine_how['percent'] = data['S2AQ6B'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
wine_how['cumulative_frequency'] = wine_how['frequency'].cumsum()
wine_how['cumulative_percent'] = wine_how['percent'].cumsum()



# ----------------------------LIQUOR------------------------------------------
liquor = pandas.DataFrame(data = data['S2AQ7A'].value_counts(sort=False, 
                          dropna = False), columns = ['frequency'])
liquor['percent'] = data['S2AQ7A'].value_counts(sort=False, 
                          normalize = True, dropna = False)*100
liquor['cumulative_frequency'] = liquor['frequency'].cumsum()
liquor['cumulative_percent'] = liquor['percent'].cumsum()

liquor_how = pandas.DataFrame(data = data['S2AQ7B'].value_counts(sort=False, 
                            dropna = False), columns = ['frequency'])
liquor_how['percent'] = data['S2AQ7B'].value_counts(sort=False, 
                            normalize = True, dropna = False)*100
liquor_how['cumulative_frequency'] = liquor_how['frequency'].cumsum()
liquor_how['cumulative_percent'] = liquor_how['percent'].cumsum()


# Show some of the results
print ("Distribution of DRANK ANY COOLERS IN LAST 12 MONTHS")
print (coolers)
print ("Distribution of HOW OFTEN DRANK COOLERS IN LAST 12 MONTHS")
print (coolers_how)