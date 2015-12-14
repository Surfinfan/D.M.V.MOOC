# -*- coding: utf-8 -*-
"""
Created on Thu Dic 05 13:39:52 2015

@author: Surfinfan

This is my work for the 'Regression Modeling in Practice' Coursera course
The question I want to study: how cultural origins influence in the way people 
chooses and uses alcoholic drinks. 

My hipothesis is that Yes, theres is a influence and different country-of-origin 
individuales will consume alcohol in a different way.

I will use the NESARC dataset considering following variables:

IDNUM    - UNIQUE ID NUMBER WITH NO ALPHABETICS
S1Q1E    - ORIGIN OR DESCENT
ETOTLCA2 - AVERAGE DAILY VOLUME OF ETHANOL CONSUMED IN PAST YEAR, FROM ALL
           TYPES OF ALCOHOLIC BEVERAGES COMBINED

"""

import pandas
import numpy
import statsmodels.formula.api as smf
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt



# Read the data from the csv file
data = pandas.read_csv('../DataSet/nesarc_pds.csv', low_memory=False)

# Let's be sure all the info is numerical or NA
# in a new DataFrame Sub that will contain all the data that I need

sub = pandas.DataFrame()
sub['country_code']  = data['S1Q1E'].convert_objects(convert_numeric = True)
sub['ethanol']       = data['ETOTLCA2'].convert_objects(convert_numeric = True)

# Chicano and Mexican-American is the same thing actually
sub['country_code']  = sub['country_code'].replace(9, 36)      

# Let's put all data with no information into NaN too
sub['country_code']  = sub['country_code'].replace(98, numpy.nan)      #other
sub['country_code']  = sub['country_code'].replace(99, numpy.nan)      # unknown


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# Now the Chi-Square analysis

# Let's create some groups of countries to analyze patterns of drink consumption

# 1: European Wine-Producing countries
# 0: Asian Countries


country_list = ['Afro-American', 
    'African',
    'Native American',
    'Australian, NZelander', 
    'Austrian',
    'Belgian',
    'Canadian',
    'Central American',
    'Chinese',
    'Cuban',
    'Czechoslovakian',
    'Danish',
    'Dutch',
    'English',
    'Filipino',
    'Finnish',
    'French',
    'German',
    'Greek',
    'Guamanian',
    'Hungarian', 
    'Indian, Afghan, Pakist.',
    'Indonesian',
    'Iranian',
    'Iraqi',
    'Irish',
    'Israeli',
    'Italian',
    'Japanese',
    'Jordanian',
    'Korean',
    'Lebanese',
    'Malaysian',
    'Mexican',
    'Mexican-American',
    'Norwegian',
    'Polish',
    'Puerto Rican',
    'Russian',
    'Scottish',
    'Samoan',
    'South American',
    'Spanish, Portugese',
    'Swedish',
    'Swiss',
    'Taiwanese',
    'Turkish',
    'Vietnamese',
    'Welsh',
    'Yugoslavian',
    'Other Asian ',
    'Caribbean (Spanish Speak)',
    'Caribbean (Non-Sp Speak)',
    'Eastern European',
    'Middle Eastern',
    'Pacific Islander',
    'Other Spanish']
    
sub['country_name'] = sub['country_code'].astype('category').dropna()
sub['country_name'] = sub['country_name'].cat.rename_categories(country_list)                         

recode_country_groups_wine = { 
    'Austrian' : 1,
    'Belgian' : 1,
    'Chinese' : 0,
    'Czechoslovakian' : 1,
    'Danish' : 1,
    'Dutch' : 1,
    'English' : 1,
    'Filipino' : 0,
    'Finnish' : 1,
    'French' : 1,
    'German' : 1,
    'Greek' : 1,
    'Hungarian' : 1, 
    'Indian, Afghan, Pakist.' : 0,
    'Indonesian' : 0,
    'Iranian' : 0,
    'Iraqi' : 0,
    'Irish' : 1,
    'Israeli' : 0,
    'Italian' : 1,
    'Japanese' : 0,
    'Jordanian' : 0,
    'Korean' : 0,
    'Lebanese' : 0,
    'Malaysian' : 0,
    'Norwegian' : 1,
    'Polish' : 1,
    'Russian' : 1,
    'Scottish' : 1,
    'Spanish, Portugese' : 1,
    'Swedish' : 1,
    'Swiss' : 1,
    'Taiwanese' : 0,
    'Turkish' : 1,
    'Vietnamese' : 0,
    'Welsh' : 1,
    'Yugoslavian': 1,
    'Other Asian ': 0,
    'Eastern European': 1,
    'Middle Eastern': 0}

sub['country_group_wine'] = sub['country_name'].map(recode_country_groups_wine)
        
# We already have a binary explanatory variable and a quanttative response
# variable (country group vs ethanol consumption

model1 = smf.ols(formula = 'ethanol ~ country_group_wine', data = sub)
result1 = model1.fit()
print(result1.summary())


