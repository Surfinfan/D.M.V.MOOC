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
ETOTLCA2 - AVERAGE DAILY VOLUME OF ETHANOL CONSUMED IN PAST YEAR, FROM ALL
           TYPES OF ALCOHOLIC BEVERAGES COMBINED
SEX      - 1 (Male) / 2 (Female)
MARITAL  - 1(Married) /  2 (Living with someone as if married) / 
           3 (Widowed) / 4 (Divorced) / 5 (Separated) / 6 (Never Married)
AGE      - 18-98. Age in years
CHLD0_17 - NUMBER OF CHILDREN UNDER AGE 18 IN HOUSEHOLD
S1Q10B   - TOTAL PERSONAL INCOME IN LAST 12 MONTHS: CATEGORY [0-17]

In this exercice I will explore the relation of marital status and childs
to the way people consumes alcohol

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
sub['ethanol'] = data['ETOTLCA2'].convert_objects(convert_numeric = True)
sub['marital'] = data ['MARITAL'].convert_objects(convert_numeric = True)
sub['sex']     = data ['SEX'].convert_objects(convert_numeric = True)

# Abstainers drink 0 ethanol a year. This is the response variable
sub['ethanol'] = sub['ethanol'].replace(numpy.nan, 0)

# Let's get categorical variables with 0 values

# marital  - 1(Married) /  2 (Living with someone as if married) / 
#            3 (Widowed) / 4 (Divorced) / 5 (Separated) / 0 (Never Married)
sub['marital'] = sub['marital'].replace(6, 0)
# sex      - 1 (Male) / 0 (Female)
sub['sex']     = sub['sex'].replace(2, 0)

        
# We already have a binary explanatory variable and a quanttative response
# variable (marital status vs ethanol consumption)

model1 = smf.ols(formula = 'ethanol ~ marital + sex', data = sub)
result1 = model1.fit()
print(result1.summary())

sub['childs']  = data ['CHLD0_17'].convert_objects(convert_numeric = True)
sub['income']  = data ['S1Q10B'].convert_objects(convert_numeric = True)
sub['age']     = data ['AGE'].convert_objects(convert_numeric = True)

# childs and income already have a 0
# Age - let's substract the minimum
sub['age']     = sub['age'] - min(sub['age'])

model2 = smf.ols(formula = 'ethanol ~ marital + sex + childs + age + income', data = sub)
result2 = model2.fit()
print(result2.summary())



