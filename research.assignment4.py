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

I will include the extra information of these columns:
S2AQ4D - NUMBER OF COOLERS USUALLY CONSUMED ON DAYS WHEN DRANK COOLERS IN 
         LAST 12 MONTHS
S2AQ5D - NUMBER OF BEERS USUALLY CONSUMED ON DAYS WHEN DRANK BEER IN 
         LAST 12 MONTHS
S2AQ6D - NUMBER OF GLASSES/CONTAINERS OF WINE USUALLY CONSUMED ON DAYS WHEN 
         DRANK WINE IN LAST 12 MONTHS
S2AQ7D - NUMBER OF DRINKS OF LIQUOR USUALLY CONSUMED ON DAYS WHEN DRANK 
         LIQUOR IN LAST 12 MONTHS

"""

import pandas
import numpy
import statsmodels.formula.api as smf
import scipy.stats


# Read the data from the csv file
data = pandas.read_csv('../DataSet/nesarc_pds.csv', low_memory=False)

# Let's be sure all the info is numerical or NA
# in a new DataFrame Sub that will contain all the data that I need

sub = pandas.DataFrame()
sub['country_code']  = data['S1Q1E'].convert_objects(convert_numeric = True)
sub['coolers_any']   = data['S2AQ4A'].convert_objects(convert_numeric = True)
sub['coolers_often'] = data['S2AQ4B'].convert_objects(convert_numeric = True)
sub['coolers_num']   = data['S2AQ4D'].convert_objects(convert_numeric = True)
sub['beers_any']     = data['S2AQ5A'].convert_objects(convert_numeric = True)
sub['beers_often']   = data['S2AQ5B'].convert_objects(convert_numeric = True)
sub['beers_num']     = data['S2AQ5D'].convert_objects(convert_numeric = True)
sub['wine_any']      = data['S2AQ6A'].convert_objects(convert_numeric = True)
sub['wine_often']    = data['S2AQ6B'].convert_objects(convert_numeric = True)
sub['wine_num']      = data['S2AQ6D'].convert_objects(convert_numeric = True)
sub['liquor_any']    = data['S2AQ7A'].convert_objects(convert_numeric = True)
sub['liquor_often']  = data['S2AQ7B'].convert_objects(convert_numeric = True)
sub['liquor_num']    = data['S2AQ7D'].convert_objects(convert_numeric = True)

# Chicano and Mexican-American is the same thing actually
sub['country_code']  = sub['country_code'].replace(9, 36)      


# Let's create a new category [0] for people that don't drink at all
sub['coolers_any']   = sub['coolers_any'].replace(numpy.nan, 0)
sub['beers_any']     = sub['beers_any'].replace(numpy.nan, 0)
sub['wine_any']      = sub['wine_any'].replace(numpy.nan, 0)
sub['liquor_any']    = sub['liquor_any'].replace(numpy.nan, 0)


# Now, let's have in consideration that most of NaNs in '_often' series
# ['_often' series of columns are the 'How often drank X ...'] are people that
# it is included in the group "I don't drink". The Codebook explains that for 
# this series NaN mean "did not drink or unknown if drank X in last 12 months" 
# This includes both sub-groups, people that don't drink and people that does  
# not know if they have drinked XXX in the last 12 months. The first sub group 
# (abstainers) are identified in the '_any' series of columns with an 0, so, 
# let's separate both sub-groups in the '_often' series. 

# Functions to determine if the row is a real abstainer or non drinker 
# It returns 0 if the individual is already considered a non-drinker

def abstainer_coolers (row):
    if (row['coolers_any'] == 0):
        return 0
    else:
        return row['coolers_often']

def abstainer_beers (row):
    if (row['beers_any'] == 0):
        return 0
    else:
        return row['beers_often']

def abstainer_wine (row):
    if (row['wine_any'] == 0):
        return 0
    else:
        return row['wine_often']

def abstainer_liquor (row):
    if (row['liquor_any'] == 0):
        return 0
    else:
        return row['liquor_often']


# and now we get for each 'B' serie the individuals considered as abstainer
# marked with a 0 and not a NaN

sub['coolers_often'] = sub.apply(lambda row:abstainer_coolers(row), axis = 1)
sub['beers_often']   = sub.apply(lambda row:abstainer_beers(row), axis = 1)
sub['wine_often']    = sub.apply(lambda row:abstainer_wine(row), axis = 1)
sub['liquor_often']  = sub.apply(lambda row:abstainer_liquor(row), axis = 1)


# Exactly the same issue with '_num' series

def abstainer_coolers_num (row):
    if (row['coolers_any'] == 0):
        return 0
    else:
        return row['coolers_num']

def abstainer_beers_num (row):
    if (row['beers_any'] == 0):
        return 0
    else:
        return row['beers_num'] 

def abstainer_wine_num (row):
    if (row['wine_any'] == 0):
        return 0
    else:
        return row['wine_num']

def abstainer_liquor_num (row):
    if (row['liquor_any'] == 0):
        return 0
    else:
        return row['liquor_num'] 


# and now we get for each 'D' serie the individuals considered as abstainer
# marked with a 0 and not a NaN

sub['coolers_num'] = sub.apply(lambda row:abstainer_coolers_num(row), axis = 1)
sub['beers_num']   = sub.apply(lambda row:abstainer_beers_num(row), axis = 1)
sub['wine_num']    = sub.apply(lambda row:abstainer_wine_num(row), axis = 1)
sub['liquor_num']  = sub.apply(lambda row:abstainer_liquor_num(row), axis = 1)


# Let's put all data with no information into NaN too
sub['country_code']     = sub['country_code'].replace(98, numpy.nan)      #other
sub['country_code']     = sub['country_code'].replace(99, numpy.nan)      # unknown
sub['coolers_any']   = sub['coolers_any'].replace(9, numpy.nan)    # unknown
sub['coolers_often'] = sub['coolers_often'].replace(99, numpy.nan)    # unknown
sub['coolers_num']   = sub['coolers_num'].replace(99, numpy.nan)    # unknown
sub['beers_any']     = sub['beers_any'].replace(9, numpy.nan)    # unknown
sub['beers_often']   = sub['beers_often'].replace(99, numpy.nan)    # unknown
sub['beers_num']     = sub['beers_num'] .replace(99, numpy.nan)    # unknown
sub['wine_any']      = sub['wine_any'].replace(9, numpy.nan)    # unknown
sub['wine_often']    = sub['wine_often'].replace(99, numpy.nan)    # unknown
sub['wine_num']      = sub['wine_num'].replace(99, numpy.nan)    # unknown
sub['liquor_any']    = sub['liquor_any'].replace(9, numpy.nan)    # unknown
sub['liquor_often']  = sub['liquor_often'].replace(99, numpy.nan)    # unknown
sub['liquor_num']    = sub['liquor_num'] .replace(99, numpy.nan)    # unknown


# In the '_any' series of data, it seems that it would be more logical to have 
# a different order of dummy codes. Now, '0' means 'Don'k drik at all', 
# '1' means 'Yes, I Drunk XXX in the last 12 months' and '2' means 'No, I'm 
# not abstainer but I did not drink XXX in thelast 12 months'
# I would prefer a distribution like:
# '0' - Abstainer
# '1' - Not abstainer, but didn't drink XXX in the last 12 months
# '2' - Yes, drunk XXX in the last 12 months
# NaN - Don't know

recodeA = {0:0, 1:2, 2:1}
sub['coolers_any'] = sub['coolers_any'].map(recodeA)
sub['beers_any']   = sub['beers_any'].map(recodeA)
sub['wine_any']    = sub['wine_any'].map(recodeA)
sub['liquor_any']  = sub['liquor_any'].map(recodeA)

# In the '_often' series, the shorter the number in the dummy code, the higher 
# the intensity of drinking. It would be more logical to invert the dummy codes
# to this new meaning:

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
sub['coolers_often'] = sub['coolers_often'].map(recodeB)
sub['beers_often']   = sub['beers_often'].map(recodeB)
sub['wine_often']    = sub['wine_often'].map(recodeB)
sub['liquor_often']  = sub['liquor_often'].map(recodeB)


# In '_num' series of data, order seems to be logical as it is
# '_num' Series give you the number of drinks (number of cups of wine, beer, 
# etc) that the individual consumes each time that drinks this kind of drink

# Let's create some new variables showing: 
# 1) how many days does individuals drink every month
# 2) how many drinks of each type does every individual drinks every month

def how_many_days (num):
    if (num == 0):
        return 0.0
    elif (num == 1):
        return 1.5/12
    elif (num == 2):
        return 4.5/12
    elif (num == 3):
        return 9/12
    elif (num == 4):
        return 1.0
    elif (num == 5):
        return 2.5
    elif (num == 6):
        return 4.0
    elif (num == 7):
        return 8.0
    elif (num == 8):
        return 3.5*4
    elif (num == 9):
        return 25.0
    elif (num == 10):
        return 30.0
    else:
        return numpy.nan


sub['days_drinking_coolers'] = sub.apply(lambda row:how_many_days(row['coolers_often']), axis = 1)
sub['days_drinking_beers']   = sub.apply(lambda row:how_many_days(row['beers_often']), axis = 1)
sub['days_drinking_wine']    = sub.apply(lambda row:how_many_days(row['wine_often']), axis = 1)
sub['days_drinking_liquor']  = sub.apply(lambda row:how_many_days(row['liquor_often']), axis = 1)

sub['how_many_coolers'] =  sub['days_drinking_coolers'] * sub['coolers_num']   
sub['how_many_beers']   =  sub['days_drinking_beers'] * sub['beers_num'] 
sub['how_many_wine']    =  sub['days_drinking_wine'] * sub['wine_num']
sub['how_many_liquor']  =  sub['days_drinking_liquor'] * sub['liquor_num']     
    

# Now there are blocks of code, basically all very similar, in order to
# show distribution of different variables of interest for me
# First in absolute value, then in %, then cumulative, then % cumulative


#----------------------------COUNTRIES OF ORIGIN-------------------------------
countries = pandas.DataFrame(data = sub['country_code'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
countries['percent'] = sub['country_code'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
countries['cumulative_frequency'] = countries['frequency'].cumsum()
countries['cumulative_percent'] = countries['percent'].cumsum()


# ----------------------------COOLERS------------------------------------------
coolers = pandas.DataFrame(data = sub['coolers_any'].value_counts(sort=False, 
                           dropna = False), columns = ['frequency'])
coolers['percent'] = sub['coolers_any'].value_counts(sort=False, 
                           normalize = True, dropna = False)*100
coolers['cumulative_frequency'] = coolers['frequency'].cumsum()
coolers['cumulative_percent'] = coolers['percent'].cumsum()

coolers_how = pandas.DataFrame(data = sub['coolers_often'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
coolers_how['percent'] = sub['coolers_often'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
coolers_how['cumulative_frequency'] = coolers_how['frequency'].cumsum()
coolers_how['cumulative_percent'] = coolers_how['percent'].cumsum()


# ----------------------------BEERS------------------------------------------
beers = pandas.DataFrame(data = sub['beers_any'].value_counts(sort=False, 
                         dropna = False), columns = ['frequency'])
beers['percent'] = sub['beers_any'].value_counts(sort=False, 
                         normalize = True, dropna = False)*100
beers['cumulative_frequency'] = beers['frequency'].cumsum()
beers['cumulative_percent'] = beers['percent'].cumsum()

beers_how = pandas.DataFrame(data = sub['beers_often'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
beers_how['percent'] = sub['beers_often'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
beers_how['cumulative_frequency'] = beers_how['frequency'].cumsum()
beers_how['cumulative_percent'] = beers_how['percent'].cumsum()


# ----------------------------WINE------------------------------------------
wine = pandas.DataFrame(data = sub['wine_any'].value_counts(sort=False, 
                        dropna = False), columns = ['frequency'])
wine['percent'] = sub['wine_any'].value_counts(sort=False, 
                        normalize = True, dropna = False)*100
wine['cumulative_frequency'] = wine['frequency'].cumsum()
wine['cumulative_percent'] = wine['percent'].cumsum()

wine_how = pandas.DataFrame(data = sub['wine_often'].value_counts(sort=False, 
                             dropna = False), columns = ['frequency'])
wine_how['percent'] = sub['wine_often'].value_counts(sort=False, 
                             normalize = True, dropna = False)*100
wine_how['cumulative_frequency'] = wine_how['frequency'].cumsum()
wine_how['cumulative_percent'] = wine_how['percent'].cumsum()



# ----------------------------LIQUOR------------------------------------------
liquor = pandas.DataFrame(data = sub['liquor_any'].value_counts(sort=False, 
                          dropna = False), columns = ['frequency'])
liquor['percent'] = sub['liquor_any'].value_counts(sort=False, 
                          normalize = True, dropna = False)*100
liquor['cumulative_frequency'] = liquor['frequency'].cumsum()
liquor['cumulative_percent'] = liquor['percent'].cumsum()

liquor_how = pandas.DataFrame(data = sub['liquor_often'].value_counts(sort=False, 
                            dropna = False), columns = ['frequency'])
liquor_how['percent'] = sub['liquor_often'].value_counts(sort=False, 
                            normalize = True, dropna = False)*100
liquor_how['cumulative_frequency'] = liquor_how['frequency'].cumsum()
liquor_how['cumulative_percent'] = liquor_how['percent'].cumsum()


# Show some of the results
print ("Distribution of DRANK ANY COOLERS IN LAST 12 MONTHS")
print (coolers)
print ("Distribution of HOW OFTEN DRANK COOLERS IN LAST 12 MONTHS")
print (coolers_how)
print ("Distribution of DRANK ANY BEERS IN LAST 12 MONTHS")
print (beers)
print ("Distribution of HOW OFTEN DRANK BEERS COOLERS IN LAST 12 MONTHS")
print (beers_how)


# show description of quantitative data:
# How many XXX drunk in the month

desc_quant = pandas.DataFrame(data = ['Num. of coolers in a month', 
                        'Num. of beers in a month', 
                        'Num of units of wine in a month',
                        'Num of units of liquor in a month',], 
                        columns = ['Data'])
desc_quant ['mean'] = [sub['how_many_coolers'].mean(), 
                       sub['how_many_beers'].mean(), 
                       sub['how_many_wine'].mean(),
                       sub['how_many_liquor'].mean()]

desc_quant ['max'] =  [sub['how_many_coolers'].max(), 
                       sub['how_many_beers'].max(), 
                       sub['how_many_wine'].max(),
                       sub['how_many_liquor'].max()]

desc_quant ['std'] =  [sub['how_many_coolers'].std(), 
                       sub['how_many_beers'].std(), 
                       sub['how_many_wine'].std(),
                       sub['how_many_liquor'].std()]

desc_quant ['median'] = [sub['how_many_coolers'].median(), 
                       sub['how_many_beers'].median(), 
                       sub['how_many_wine'].median(),
                       sub['how_many_liquor'].median()]
                       
print (desc_quant)
              
# --------------------------------------------------------------------------

# Now some inferential statistics

# I will use 'S1Q1F' ('BORN IN THE USA?' [1=YES/2=NO/9=UNKNOWN])
# To try to determine if people not born in the USA have the same 
# behaviour than people born in the USA

recodeBorn      = {1:1, 2:0, 9:numpy.nan}
sub['usa_born'] = data['S1Q1F'].convert_objects(convert_numeric = True)
sub['usa_born'] = sub['usa_born'].map(recodeBorn)
sub['usa_born'] = sub['usa_born'].astype('category').dropna()
sub_coolers     = sub[['usa_born', 'how_many_coolers']]
sub_beers       = sub[['usa_born', 'how_many_beers']]
sub_wine        = sub[['usa_born', 'how_many_wine']]
sub_liquor      = sub[['usa_born', 'how_many_liquor']]


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# Now the ANOVA analysis
# Let's see the behaviour for coolers
model1 = smf.ols(formula = 'how_many_coolers ~ C(usa_born)', data = sub_coolers)
result1 = model1.fit()
print(result1.summary())

m_coolers = sub_coolers.groupby('usa_born').mean()
s_coolers = sub_coolers.groupby('usa_born').std()

print (m_coolers)
print (s_coolers)

# --------------------------------------------------------------------------

# Let's see the behaviour for beers
model2 = smf.ols(formula = 'how_many_beers ~ C(usa_born)', data = sub_beers)
result2 = model2.fit()
print(result2.summary())

m_beers = sub_beers.groupby('usa_born').mean()
s_beers = sub_beers.groupby('usa_born').std()

print (m_beers)
print (s_beers)


# --------------------------------------------------------------------------

# Let's see the behaviour for wine
model3 = smf.ols(formula = 'how_many_wine ~ C(usa_born)', data = sub_wine)
result3 = model3.fit()
print(result3.summary())

m_wine = sub_wine.groupby('usa_born').mean()
s_wine = sub_wine.groupby('usa_born').std()

print (m_wine)
print (s_wine)


# --------------------------------------------------------------------------

# Let's see the behaviour for liquor
model4 = smf.ols(formula = 'how_many_liquor ~ C(usa_born)', data = sub_liquor)
result4 = model4.fit()
print(result4.summary())

m_liquor = sub_liquor.groupby('usa_born').mean()
s_liquor = sub_liquor.groupby('usa_born').std()

print (m_liquor)
print (s_liquor)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# Now the Chi-Square analysis

# Let's create some groups of countries to analyze patterns of drink consumption

# 1: European Wine-Producing countries
# 2: Other european countries
# 3: Other Wine-producing countries
# 4: Near and middle-east countries
# 5: Asian Countries
# 6: Latin Countries
# 0: Others


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

recode_country_groups_wine = {'Afro-American' : 0, 
    'African' : 0,
    'Native American' : 0,
    'Australian, NZelander' : 3, 
    'Austrian' : 1,
    'Belgian' : 2,
    'Canadian' : 0,
    'Central American' : 6,
    'Chinese' : 3,
    'Cuban' : 6,
    'Czechoslovakian' : 2,
    'Danish' : 2,
    'Dutch' : 2,
    'English' : 2,
    'Filipino' : 5,
    'Finnish' : 2,
    'French' : 1,
    'German' : 1,
    'Greek' : 1,
    'Guamanian' : 0,
    'Hungarian' : 1, 
    'Indian, Afghan, Pakist.' : 5,
    'Indonesian' : 5,
    'Iranian' : 4,
    'Iraqi' : 4,
    'Irish' : 2,
    'Israeli' : 4,
    'Italian' : 1,
    'Japanese' : 5,
    'Jordanian' : 4,
    'Korean' : 5,
    'Lebanese' : 4,
    'Malaysian' : 5,
    'Mexican' : 6,
    'Mexican-American' : 6,
    'Norwegian' : 2,
    'Polish' : 2,
    'Puerto Rican' : 6,
    'Russian' : 1,
    'Scottish' : 2,
    'Samoan' : 0,
    'South American': 3,
    'Spanish, Portugese' : 1,
    'Swedish' : 2,
    'Swiss' : 2,
    'Taiwanese' : 5,
    'Turkish' : 4,
    'Vietnamese' : 5,
    'Welsh' : 2,
    'Yugoslavian': 1,
    'Other Asian ': 5,
    'Caribbean (Spanish Speak)': 6,
    'Caribbean (Non-Sp Speak)': 0,
    'Eastern European': 2,
    'Middle Eastern': 4,
    'Pacific Islander': 2,
    'Other Spanish': 6}

recode_groups_names_wine = { 1: 'European Wine-Producing countries',
                      2: 'Other european countries'
                      3: 'Other Wine-producing countries'
                      4: 'Near and middle-east countries'
                      5: 'Asian Countries'
                      6: 'Latin Countries'
                      0: 'Others'
}
    
sub['country_group_wine'] = sub['country_name'].map(recode_country_groups_wine)
sub['country_nameg_wine'] = sub['country_group_wine'].map(recode_groups_names_wine)


# Let's remap the '2' in the '_any' variables to 1 to get the info we want 
# to study, and do the same process to each type of drinking

recodeDrink = {0:0, 1:0, 2:1}

sub['wine_drinkers'] = sub['wine_any'].map(recodeDrink)


# Now the Chi-square test:
# Explanatory Variable: country_group_wine
# Response Variable: wine_drinkers

pandas.crosstab(sub['wine_drinkers'], sub['country_group_wine'])
ct1 = pandas.crosstab(sub['wine_drinkers'], sub['country_group_wine'])
colsum = ct1.sum(axis=0)
colpct = ct1/colsum
cs1 = scipy.stats.chi2_contingency(ct1)


# Now we try to get the different pvalues of all pairwise comparision
# We do it in a for bucle creating a new dataframe PxP to view all chi2 
# values and all pvalues for each pair

# We create a couple of empty square tables tables to store results
chi2 = pandas.DataFrame(numpy.random.randn(ct1.axes[1].max()+1, 
                                           ct1.axes[1].max()+1))
chi2.ix[:] = numpy.nan
pvalue = pandas.DataFrame(numpy.random.randn(ct1.axes[1].max()+1, 
                                             ct1.axes[1].max()+1))
pvalue.ix[:] = numpy.nan

num_comp = 0

# Now the bucle, but it only calculates chi2 and pvalue if the pair has not 
# been already calculated

for ax1 in ct1.axes[1]:
    for ax2 in ct1.axes[1]:
        if ax1 == ax2:
            continue
        ax1 = ax1.astype(numpy.int64)
        ax2 = ax2.astype(numpy.int64)        
        if not(numpy.isnan(chi2[ax2][ax1])):
            continue
        recode_chi = {ax1:ax1, ax2:ax2}
        versus = ax1.astype('str') + 'v' + ax2.astype('str')
        sub[versus] = sub['country_group_wine'].map(recode_chi)
        ct2 = pandas.crosstab(sub['wine_drinkers'], sub[versus])
        cs2 = scipy.stats.chi2_contingency(ct2)
        chi2[ax1][ax2] = cs2[0]
        pvalue[ax1][ax2] = cs2[1]
        num_comp += 1
        
        
# We must use the Bonferroni Adjustment

thresold_bonberroni = 0.05 / num_comp
rejected_h0 = pvalue < thresold_bonberroni

print rejected_h0

        
        
        