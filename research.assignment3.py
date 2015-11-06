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
import seaborn as sns
import matplotlib.pyplot as plt

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
# Now some graphics!

# First graphic: Distribution of individuals per country of origin
# I prefer to use name of countries-origin instead of codes

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

# I get the correct order to display in the graphic (from more to less)
country_order = pandas.DataFrame(data = sub['country_code'].value_counts(sort=False, 
                             dropna = True), columns = ['frequency'])
country_order['name'] = country_list
country_order = country_order.sort(columns = 'frequency', ascending = False)
sns.countplot(y='country_name', data = sub, order=country_order['name'])

plt.title('Country-origin Distribution')
plt.ylabel('Country')
plt.xlabel('# individuals')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure1_country_distribution")
plt.close()


# --------------------------------------------------------------------------
# Second graphic: Distribution of How many beers drunk in a month

# I get the correct order to display in the graphic (from more to less)
how_many_order = pandas.DataFrame(sub['how_many_beers'].value_counts(sort=False),
                                  columns=['howmany'])
how_many_order = how_many_order.sort_index()
sns.countplot(y='how_many_beers', data = sub, order = how_many_order.index)
plt.title('How many beers in a month Distribution [with 0 included]')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure2_how_many_beer_with_0")
plt.close()

# There are a lot of values '0', masking all the other values in the graphic
# So let's see all the others just to see it better

sns.countplot(y='how_many_beers', data = sub[sub.how_many_beers!=0], 
              order=how_many_order.index[1:])
plt.title('How many beers in a month Distribution [without 0 included]')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure3_how_many_beer_with_0")
plt.close()


# As "how many XXX" gets a large set of values, I will create new categories
# for easier understading. I will use the same limits for categories in every
# type of drink although different limits would be more adecuate since it is 
# not the same to drink 5 beers than 5 stronger liquors
# 
# 0 drinks - don't drink
# (0, 4]    drinks - occasional drinker
# (4, 10]   drinks - social drinker
# (10, 60]  drinks - big drinker
# > 60      drinks - heavy drinker

sub['coolershowmany_group'] = pandas.cut(sub.how_many_coolers, 
                            [-0.1,0,4,10,60,sub.how_many_coolers.max()])
sub['beershowmany_group'] = pandas.cut(sub.how_many_beers, 
                            [-0.1,0,4,10,60,sub.how_many_beers.max()])
sub['winehowmany_group'] = pandas.cut(sub.how_many_wine, 
                            [-0.1,0,4,10,60,sub.how_many_wine.max()])
sub['liquorhowmany_group'] = pandas.cut(sub.how_many_liquor, 
                            [-0.1,0,4,10,60,sub.how_many_liquor.max()])


# Let's see them as groups, but let's try sopme subplotting

drinks = ['coolers', 'beers', 'wine', 'liquor']
fig, our_axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), sharey=True)
drinks_axes = [ax for ax_row in our_axes for ax in ax_row]      

for i, drink in enumerate(drinks):
    current_ax = drinks_axes[i]
    
    current_variable = drink+'howmany_group'
    #x_axis = current_drink_data.values.categories
    sns.countplot(x=current_variable, data = sub, ax = current_ax)
    current_ax.set_title(drink)         # Give our Axes a unique title
    current_ax.set_ylabel("#individuals")
    current_ax.set_xlabel("How many drinks ("+drink+") at month")

fig.suptitle('How many drinks a month per type of drink', fontsize=22)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure4_Groups_of_how_many_drinks_a_month")
plt.close() 


# --------------------------------------------------------------------------
# Now some cross-variables plots

# First let's explore the question:
# how country of origin influentiate in abstainers?

# Abstainers are considered in all '_any' variables, codede as '0'

recodeAbs = {0:1, 1:0, 2:0}
sub['abstainers']     = sub['coolers_any'].map(recodeAbs)
sub['country_code_a'] = sub['country_code']*sub['abstainers']
# Lets's get some order
countries_total       = sub['country_code'].value_counts(sort=False).sort_index()
countries_abstainer   = sub['country_code_a'].value_counts(sort=False).sort_index()
countries_abstainer   = countries_abstainer[1:]
abstainer_order       = pandas.DataFrame(countries_abstainer/countries_total,
                                         columns=['abstainers_percent'])
abstainer_order['name'] = country_list
abstainer_order       = abstainer_order.sort(columns = 'abstainers_percent',
                                             ascending = False)
sns.factorplot(y='country_name', x='abstainers', data=sub, kind='bar', ci=None,
               order = abstainer_order['name'])
plt.title('Abstainers by country of origin')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure5_abstainers_by_country")
plt.close()               


# Another question: how country of origin influentiate consuming each type 
# of drink

# Let's remap the '2' in the '_any' variables to 1 to get the info we want 
# to study, and do the same process to each type of drinking

recodeDrink = {0:0, 1:0, 2:1}
sub['cooler_drinkers']    = sub['coolers_any'].map(recodeDrink)
sub['beer_drinkers']      = sub['beers_any'].map(recodeDrink)
sub['wine_drinkers']      = sub['wine_any'].map(recodeDrink)
sub['liquor_drinkers']    = sub['liquor_any'].map(recodeDrink)
sub['country_code_co']    = sub['country_code']*sub['cooler_drinkers']
sub['country_code_be']    = sub['country_code']*sub['beer_drinkers']
sub['country_code_wi']    = sub['country_code']*sub['wine_drinkers']
sub['country_code_li']    = sub['country_code']*sub['liquor_drinkers']
countries_cooler_drinkers = sub['country_code_co'].value_counts(sort=False).sort_index()
countries_cooler_drinkers = countries_cooler_drinkers[1:]
countries_beer_drinkers   = sub['country_code_be'].value_counts(sort=False).sort_index()
countries_beer_drinkers   = countries_beer_drinkers[1:]
countries_wine_drinkers   = sub['country_code_wi'].value_counts(sort=False).sort_index()
countries_wine_drinkers   = countries_wine_drinkers[1:]
countries_liquor_drinkers = sub['country_code_li'].value_counts(sort=False).sort_index()
countries_liquor_drinkers = countries_liquor_drinkers[1:]
cooler_drinkers_order     = pandas.DataFrame(countries_cooler_drinkers/countries_total,
                                  columns=['cooler_drinkers_percent'])
beer_drinkers_order       = pandas.DataFrame(countries_beer_drinkers/countries_total,
                                  columns=['beer_drinkers_percent'])
wine_drinkers_order       = pandas.DataFrame(countries_wine_drinkers/countries_total,
                                  columns=['wine_drinkers_percent'])
liquor_drinkers_order     = pandas.DataFrame(countries_liquor_drinkers/countries_total,
                                  columns=['liquor_drinkers_percent'])
cooler_drinkers_order['name'] = country_list                                
beer_drinkers_order['name'] = country_list                                
wine_drinkers_order['name'] = country_list                                
liquor_drinkers_order['name'] = country_list                                
cooler_drinkers_order     = cooler_drinkers_order.sort(columns = 'cooler_drinkers_percent',
                                             ascending = False)
beer_drinkers_order       = beer_drinkers_order.sort(columns = 'beer_drinkers_percent',
                                             ascending = False)
wine_drinkers_order       = wine_drinkers_order.sort(columns = 'wine_drinkers_percent',
                                             ascending = False)
liquor_drinkers_order     = liquor_drinkers_order.sort(columns = 'liquor_drinkers_percent',
                                             ascending = False)


sns.factorplot(y='country_name', x='cooler_drinkers', data=sub, kind='bar', ci=None,
               order = cooler_drinkers_order['name'])
plt.title('Cooler drinkers by country of origin')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure6_cooler_drinkers_by_country")
plt.close()                                                 
                                             
                                  
sns.factorplot(y='country_name', x='beer_drinkers', data=sub, kind='bar', ci=None,
               order = beer_drinkers_order['name'])
plt.title('Beer drinkers by country of origin')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure7_beer_drinkers_by_country")
plt.close()                                        
                                
                                  
sns.factorplot(y='country_name', x='wine_drinkers', data=sub, kind='bar', ci=None,
               order = wine_drinkers_order['name'])
plt.title('Wine drinkers by country of origin')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure8_wine_drinkers_by_country")
plt.close()      


sns.factorplot(y='country_name', x='liquor_drinkers', data=sub, kind='bar', ci=None,
               order = liquor_drinkers_order['name'])
plt.title('Liquor drinkers by country of origin')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig("figure9_liquor_drinkers_by_country")
plt.close()      


