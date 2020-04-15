#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CHRISTINA JOHN GEORGE 
# MSBA-5


# INTRODUCTION

# Apprentice Chef, Inc. is a unique compnay that offers precooked meals that can be main within 30 minutes to their customers .It was a way that would allow working class consumers to feel as though they were cooking their whole meal. 
# However, After three years serving customers across the San Francisco Bay Area, the executives at Apprentice Chef have come to realize that over 90% of their revenue comes from customers that have been ordering meal sets for 12 months or less. 
# Given this , the following machine learning model goes deep into their strong point and also a few recommendations and what they could do next .

# In[2]:


# importing required libraries
import pandas                       as pd 
import matplotlib.pyplot            as plt 
import seaborn                      as sns 
import random                       as rand 
import statsmodels.formula.api      as smf
import gender_guesser.detector      as gender
from sklearn.model_selection        import train_test_split  
from sklearn.linear_model           import LinearRegression


# Importing dataset

# In[3]:


#  file
file = 'Apprentice_Chef_Dataset.xlsx'


# reading the file 
original_df = pd.read_excel(file)


# FEATURE ENGINEERING

# In[4]:


original_df[original_df['AVG_TIME_PER_SITE_VISIT'] >100]


# *In the above dataframe we can see that when people spend less time attending master classes and watching the videos, they are most likely NOT to Cross selll to the premium plan .
# *There are customers who have not viewed any photos of the website but followed recommendations and ordered many meals.

# In[5]:


original_df[original_df['TOTAL_MEALS_ORDERED']>20]


# In the above data frame you can see ,people who have high total orders have only a few unquie purchases and people who have less total orders have many unique purchases . This indicates that people who have high orders and less unquie purchases are customers who are used to certain meals and the ones who have more unquie purchases like to experiment with their food.

# In[6]:


print(f"""
VARIABLE NAME
-------------
{original_df['PRODUCT_CATEGORIES_VIEWED'].value_counts().sort_index()}


VARIABLE NAME
-------------
{original_df['TASTES_AND_PREFERENCES'].value_counts().sort_index()}



VARIABLE NAME
-------------
{original_df['FOLLOWED_RECOMMENDATIONS_PCT'].value_counts().sort_index()}



VARIABLE NAME
-------------
{original_df['MEDIAN_MEAL_RATING'].value_counts().sort_index()}



VARIABLE NAME
-------------
{original_df['WEEKLY_PLAN'].value_counts().sort_index()}
      
VARIABLE NAME
-------------
{original_df['MOBILE_NUMBER'].value_counts().sort_index()}      
""")


# Outlier Analysis 

# In this step we are Setting Thresholds.

# In[7]:


#Variable (Response) 
REVENUE_HI                     = 4500

#Outlier Boundaries 
TOTAL_MEALS_ORDERED_HI         = 300
UNIQUE_MEALS_PURC_HI           = 11
CONTACTS_W_CUSTOMER_SERVICE_LO = 2.7
CONTACTS_W_CUSTOMER_SERVICE_HI = 11 
AVG_TIME_SPENT_PER_VISIT_HI    = 300
CANCELLATIONS_BEFORE_NOON_HI   = 8
CANCELLATIONS_AFTER_NOON_HI    = 2.5 
PC_LOGINS_LO                  = 4.0
PC_LOGINS_HI                   = 6.3
MOBILE_LOGINS_HI               = 2
MOBILE_LOGINS_LO               = 1
WEEKLY_PLAN_HI                 = 48
EARLY_DELIVERIES_HI            = 8
LATE_DELIVERIES_HI             = 10
AVG_PREP_VID_TIME_HI           = 350
LARGEST_ORDER_SIZE_HI          = 10
MASTER_CLASSES_ATTENDED_HI     = 3
MEDIAN_MEAL_RATING_HI          = 4
AVG_CLICKS_PER_VISIT_LO        = 6
AVG_CLICKS_PER_VISIT_HI        = 18
TOTAL_PHOTOS_VIEWED_HI         = 1000


# Feature Engineering (outlier thresholds)

# In[8]:


# Total Meals Ordered
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)


# Unique Meals Purchased
original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURC_HI]

original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)
# Contact With Customer Service 
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]
condition_lo = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_LO]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                      value      = 1,
                                      inplace    = True)

# Average Time Per Site Visit
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_SPENT_PER_VISIT_HI]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)

# Cancellations Before Noon
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] >CANCELLATIONS_BEFORE_NOON_HI]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)

# Cancellations After Noon
original_df['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

original_df['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)

# PC Logins
original_df['out_PC_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_HI]
condition_lo = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_LO]

original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# Mobile Logins
original_df['out_MOBILE_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_HI]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_LO]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)
# Weekly Plan
original_df['out_WEEKLY_PLAN'] = 0
condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]

original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)

# Early Deliveries 
original_df['out_EARLY_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)
# Late Deliveries

original_df['out_LATE_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


# Average Prep Video Time
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]
 
original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)
# Largest Order Size
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


# Master Classes Attended
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)



# Median Meal Rating

original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_HI]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)
# Average Clicks Per Visit
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_LO]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)
original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                     value      = 1,
                                     inplace    = True)
# Total Photos Viewed
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > AVG_CLICKS_PER_VISIT_HI]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)


# Ploting graphs 

# In[9]:


########################
# Visual EDA (Scatterplots)
########################

plt.subplot(2, 2, 2)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['TOTAL_MEALS_ORDERED'],
                color = 'y')
plt.xlabel('TOTAL_MEALS_ORDERED')
 
##########################################    
fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['UNIQUE_MEALS_PURCH'],
                color = 'g')
plt.xlabel('UNIQUE_MEALS_PURCH')

########################

########################

plt.subplot(2, 2, 3)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['CONTACTS_W_CUSTOMER_SERVICE'],
                color = 'orange')
plt.xlabel('CONTACTS_W_CUSTOMER_SERVICE')

########################
plt.subplot(2, 2, 4)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['AVG_TIME_PER_SITE_VISIT'],
                color = 'r')
plt.xlabel('AVG_TIME_PER_SITE_VISIT')
plt.tight_layout()
#plt.savefig('Housing Data Scatterplots 1 of 5.png')
plt.show()

########################
########################

fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['CANCELLATIONS_BEFORE_NOON'],
                color = 'g')
plt.xlabel('CANCELLATIONS_BEFORE_NOON')

########################

plt.subplot(2, 2, 2)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['CANCELLATIONS_AFTER_NOON'],
                color = 'y')
plt.xlabel('CANCELLATIONS_AFTER_NOON')
############################

plt.subplot(2, 2, 3)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['PC_LOGINS'],
                color = 'orange')
plt.xlabel('PC_LOGINS')
########################

plt.subplot(2, 2, 3)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['MOBILE_LOGINS'],
                color = 'orange')
plt.xlabel('MOBILE_LOGINS')

########################

plt.subplot(2, 2, 4)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['WEEKLY_PLAN'],
                color = 'r')
plt.xlabel('WEEKLY_PLAN')
plt.tight_layout()
#plt.savefig('Housing Data Scatterplots 2 of 5.png')
plt.show()

########################
########################
fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['EARLY_DELIVERIES'],
                color = 'y')
plt.xlabel('EARLY_DELIVERIES')

########################

fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 2)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['LATE_DELIVERIES'],
                color = 'y')
plt.xlabel('LATE_DELIVERIES')

########################

plt.subplot(2, 2, 3)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['AVG_PREP_VID_TIME'],
                color = 'orange')
plt.xlabel('AVG_PREP_VID_TIME')

########################

plt.subplot(2, 2, 4)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['LARGEST_ORDER_SIZE'],
                color = 'r')
plt.xlabel('LARGEST_ORDER_SIZE')

########################

plt.subplot(2, 2, 1)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['MASTER_CLASSES_ATTENDED'],
                color = 'g')
plt.xlabel('MASTER_CLASSES_ATTENDED')
plt.tight_layout()
#plt.savefig('Housing Data Scatterplots 3 of 5.png')
plt.show()

########################
########################


fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 3)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['MEDIAN_MEAL_RATING'],
                color = 'y')
plt.xlabel('MEDIAN_MEAL_RATING')

########################
fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 2)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['AVG_CLICKS_PER_VISIT'],
                color = 'y')
plt.xlabel('AVG_CLICKS_PER_VISIT')

########################
fig, ax = plt.subplots(figsize = (10, 8))
plt.subplot(2, 2, 4)
sns.scatterplot(y = original_df['REVENUE'],
                x = original_df['TOTAL_PHOTOS_VIEWED'],
                color = 'y')
plt.xlabel('TOTAL_PHOTOS_VIEWED')


# In this step we are trend based thresholds to identify where a variable's trend changes in terms of its relationship with what we are trying to predict (Revenue).

# In[10]:


TOTAL_MEALS_ORDERED_CHANGE_LO         = 20
UNIQUE_MEALS_PURCH_CHANGE_1           = 1 
AVG_TIME_PER_SITE_VISIT_CHANGE_LO     = 100
CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI = 10 
CANCELLATIONS_BEFORE_NOON_CHANGE_1   = 0
CANCELLATIONS_AFTER_NOON_CHANGE_1    = 0
WEEKLY_PLAN_CHANGE_1                = 0
MASTER_CLASSES_ATTENDED_CHANGE_LO     = 1
AVG_PREP_VID_TIME_CHANGE_HI           = 250
LARGEST_ORDER_SIZE_CHANGE_HI          = 5
MEDIAN_MEAL_RATING_CHANGE_1         = 4
AVG_CLICKS_PER_VISIT_CHANGE_HI        = 15
TOTAL_PHOTOS_VIEWED_CHANGE_1         = 0


# In[11]:


# total meals ordered
original_df['change_TOTAL_MEALS_ORDERED'] = 0
condition = original_df.loc[0:,'change_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_CHANGE_LO ]

original_df['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


# Average Time Spent Per Visit
original_df['change_AVG_TIME_PER_SITE_VISIT'] = 0

condition = original_df.loc[0:,'change_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] < AVG_TIME_PER_SITE_VISIT_CHANGE_LO]

original_df['change_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)



# Contact With Customer Service 
original_df['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = original_df.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI]

original_df['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)


# avg click per vist 
original_df['change_AVG_CLICKS_PER_VISIT'] = 0
condition = original_df.loc[0:,'change_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_CHANGE_HI ]

original_df['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Average Prep Vid
original_df['change_AVG_PREP_VID_TIME'] = 0
condition = original_df.loc[0:,'change_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI ]

original_df['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# largest order size
original_df['change_LARGEST_ORDER_SIZE'] = 0
condition = original_df.loc[0:,'change_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_CHANGE_HI ]

original_df['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# Master Classes Attended
original_df['change_MASTER_CLASSES_ATTENDED'] = 0
condition = original_df.loc[0:,'change_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] < MASTER_CLASSES_ATTENDED_CHANGE_LO ]

original_df['change_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# total meals ordered
original_df['change_TOTAL_MEALS_ORDERED'] = 0
condition = original_df.loc[0:,'change_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_CHANGE_LO ]

original_df['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

########################################
## change threshold                ##
########################################


# unique meals 
original_df['change_UNIQUE_MEALS_PURCH'] = 0
condition = original_df.loc[0:,'change_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] == UNIQUE_MEALS_PURCH_CHANGE_1]

original_df['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# total photos viewed 
original_df['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = original_df.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_CHANGE_1]

original_df['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# cancellations before noon
original_df['change_CANCELLATIONS_BEFORE_NOON'] = 0
condition = original_df.loc[0:,'change_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] == CANCELLATIONS_BEFORE_NOON_CHANGE_1]

original_df['change_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                  value      = 1,
                                  inplace    = True)


# cancellations afternoon
original_df['change_CANCELLATIONS_AFTER_NOON'] = 0
condition = original_df.loc[0:,'change_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] == CANCELLATIONS_AFTER_NOON_CHANGE_1]

original_df['change_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                  value      = 1,
                                  inplace    = True)


# Weekly plan
original_df['change_WEEKLY_PLAN'] = 0
condition = original_df.loc[0:,'change_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] == WEEKLY_PLAN_CHANGE_1]

original_df['change_WEEKLY_PLAN'].replace(to_replace = condition,
                                  value      = 1,
                                  inplace    = True)



# Median meal rating 
original_df['change_MEDIAN_MEAL_RATING'] = 0
condition = original_df.loc[0:,'change_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_CHANGE_1]

original_df['change_MEDIAN_MEAL_RATING'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)


# Creating box plot for revenue and followed recommendations pct 

# In[12]:


def categorical_boxplots(response,cat_var,data):
    """
	This function can be used for categorical variables

	PARAMETERS
	----------
	response : str, response variable
	cat_var  : str, categorical variable
	data     : DataFrame of the response and categorical variables
	"""

    data.boxplot(column 	     = response,
    	            by 	 		 = cat_var,
        	        vert 	     = False,
            	    patch_artist = False,
                	meanline     = True,
               		showmeans    = True)

    plt.suptitle("")
    plt.show()


# calling the function for each categorical variable
categorical_boxplots(response = "REVENUE",
					 cat_var  = "FOLLOWED_RECOMMENDATIONS_PCT",
					 data     = original_df)


# Correlation Analysis 

# In[13]:


# correlation heatmap

fig, ax = plt.subplots(figsize=(17,17))
original_df_corr = original_df.corr().round(2)
original_df_corr2 = original_df_corr.iloc[1:21, 1:21]

sns.heatmap(original_df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('chef Correlation Heatmap.png')
plt.show()


# *Cross sell has a high correlation with followed recommendation which states that people who follow recommendations have upgraded to their premium plan .
# 

# In[14]:


# creating a (Pearson) correlation matrix
original_df_corr = original_df.corr().round(2)


# printing (Pearson) correlations with SalePrice
print(original_df_corr.loc['REVENUE'].sort_values(ascending = False))


# Explanatory Data analysis.

# In[15]:


original_df_explanatory = original_df.copy()


# dropping SalePrice and Order from the explanatory variable set
original_df_explanatory = original_df_explanatory.drop('REVENUE', axis = 1)


# formatting each explanatory variable for statsmodels
for val in original_df_explanatory:
    print(f"original_df{[val]} +")


# Sort emails to divide the emails into segments .

# In[16]:


emailsort_lst = []

# looping over each email address
for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    emailsort_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(emailsort_lst)


# displaying the results
#email_df
email_df[1].value_counts()


# In[17]:


personal_email_domains      = ['@gmail.com',
                                  '@yahoo.com',
                                  '@protonmail.com']

professional_email_domains  = ['@mmm.com',
                               '@amex.com',
                               '@apple.com',
                               '@boeing.com',
                               '@caterpillar.com',
                               '@chevron.com',
                               '@cisco.com',
                               '@cocacola.com',
                               '@disney.com',
                               '@dupont.com',
                               '@exxon.com',
                               '@ge.org',
                               '@goldmansacs.com',
                               '@homedepot.com',
                               '@ibm.com',
                               '@intel.com',
                               '@jnj.com',
                               '@jpmorgan.com',
                               '@mcdonalds.com',
                               '@merck.com',
                               '@microsoft.com',
                               '@nike.com',
                               '@pfizer.com',
                               '@pg.com',
                               '@travelers.com',
                               '@unitedtech.com',
                               '@unitedhealth.com',
                               '@verizon.com',
                               '@visa.com',
                               '@walmart.com']

junk_email_domains       = [
                            '@me.com',
                            '@aol.com',
                            '@hotmail.com',
                            '@live.com',
                            '@msn.com',
                            '@passport.com'
                            ]



# placeholder list
emailsort_lst = []


# looping to group observations by domain type
for domain in email_df[1]:
        if '@' + domain in personal_email_domains:
            emailsort_lst.append('personal')
            
        elif '@' + domain in professional_email_domains:
            emailsort_lst.append('professional')
           
        elif '@'+ domain in junk_email_domains  :
            emailsort_lst.append('junk')
            
        else:
            print('Unknown')


# concatenating with original DataFrame
original_df['domain_group'] = pd.Series(emailsort_lst)


# checking results
original_df['domain_group'].value_counts()


# In[18]:


one_hot_Email             = pd.get_dummies(original_df['domain_group'])
original_df               = original_df.drop('EMAIL', axis = 1)
original_df               = original_df.drop('domain_group', axis = 1)
original_df               = original_df.join([one_hot_Email])


# Making a copy.

# In[19]:


original_df_explanatory_1 = original_df.copy()


# dropping SalePrice and Order from the explanatory variable set
original_df_explanatory_1 = original_df_explanatory_1.drop('REVENUE', axis = 1)


# formatting each explanatory variable for statsmodels
for val in original_df_explanatory_1:
    print(f"original_df{[val]} +")


# In[20]:


lm_full_2 = smf.ols(formula = """ original_df['REVENUE']~ original_df['CROSS_SELL_SUCCESS'] +
original_df['NAME'] +
original_df['FIRST_NAME'] +
original_df['FAMILY_NAME'] +
original_df['TOTAL_MEALS_ORDERED'] +
original_df['UNIQUE_MEALS_PURCH'] +
original_df['CONTACTS_W_CUSTOMER_SERVICE'] +
original_df['PRODUCT_CATEGORIES_VIEWED'] +
original_df['AVG_TIME_PER_SITE_VISIT'] +
original_df['MOBILE_NUMBER'] +
original_df['CANCELLATIONS_BEFORE_NOON'] +
original_df['CANCELLATIONS_AFTER_NOON'] +
original_df['TASTES_AND_PREFERENCES'] +
original_df['PC_LOGINS'] +
original_df['MOBILE_LOGINS'] +
original_df['WEEKLY_PLAN'] +
original_df['EARLY_DELIVERIES'] +
original_df['LATE_DELIVERIES'] +
original_df['PACKAGE_LOCKER'] +
original_df['REFRIGERATED_LOCKER'] +
original_df['FOLLOWED_RECOMMENDATIONS_PCT'] +
original_df['AVG_PREP_VID_TIME'] +
original_df['LARGEST_ORDER_SIZE'] +
original_df['MASTER_CLASSES_ATTENDED'] +
original_df['MEDIAN_MEAL_RATING'] +
original_df['AVG_CLICKS_PER_VISIT'] +
original_df['TOTAL_PHOTOS_VIEWED'] +
original_df['out_TOTAL_MEALS_ORDERED'] +
original_df['out_UNIQUE_MEALS_PURCH'] +
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] +
original_df['out_AVG_TIME_PER_SITE_VISIT'] +
original_df['out_CANCELLATIONS_BEFORE_NOON'] +
original_df['out_CANCELLATIONS_AFTER_NOON'] +
original_df['out_PC_LOGINS'] +
original_df['out_MOBILE_LOGINS'] +
original_df['out_WEEKLY_PLAN'] +
original_df['out_EARLY_DELIVERIES'] +
original_df['out_LATE_DELIVERIES'] +
original_df['out_AVG_PREP_VID_TIME'] +
original_df['out_LARGEST_ORDER_SIZE'] +
original_df['out_MASTER_CLASSES_ATTENDED'] +
original_df['out_MEDIAN_MEAL_RATING'] +
original_df['out_AVG_CLICKS_PER_VISIT'] +
original_df['out_TOTAL_PHOTOS_VIEWED'] +
original_df['change_TOTAL_MEALS_ORDERED'] +
original_df['change_AVG_TIME_PER_SITE_VISIT'] +
original_df['change_CONTACTS_W_CUSTOMER_SERVICE'] +
original_df['change_AVG_CLICKS_PER_VISIT'] +
original_df['change_AVG_PREP_VID_TIME'] +
original_df['change_LARGEST_ORDER_SIZE'] +
original_df['change_MASTER_CLASSES_ATTENDED'] +
original_df['change_UNIQUE_MEALS_PURCH'] +
original_df['change_TOTAL_PHOTOS_VIEWED'] +
original_df['change_CANCELLATIONS_BEFORE_NOON'] +
original_df['change_CANCELLATIONS_AFTER_NOON'] +
original_df['change_WEEKLY_PLAN'] +
original_df['change_MEDIAN_MEAL_RATING'] +
original_df['junk'] +
original_df['personal'] +
original_df['professional']  
""",data = original_df)

# telling Python to run the data through the blueprint
results_full_2 = lm_full_2.fit()


# printing the results
results_full_2.summary()


# TRAIN / TEST SPLIT

# In[21]:


# preparing explanatory variable data
original_df_data   = original_df.drop([ 'REVENUE',
                          'FIRST_NAME',
                          'FAMILY_NAME'],
                               axis = 1)


# preparing response variable data
original_df_target = original_df.loc[:, 'REVENUE']


# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222)


# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# FINAL MODEL 

# ORDINARY LEAST SQUARE REGRESSION 

# In[22]:


x_variables = ['CROSS_SELL_SUCCESS',
'TOTAL_MEALS_ORDERED',
'CONTACTS_W_CUSTOMER_SERVICE',
'LATE_DELIVERIES',
'AVG_PREP_VID_TIME',
'LARGEST_ORDER_SIZE',
'MASTER_CLASSES_ATTENDED',
'MEDIAN_MEAL_RATING',
'TOTAL_PHOTOS_VIEWED',
'out_AVG_PREP_VID_TIME',
'out_MASTER_CLASSES_ATTENDED',
'change_CONTACTS_W_CUSTOMER_SERVICE',
'change_AVG_PREP_VID_TIME',
'change_UNIQUE_MEALS_PURCH',
'change_MEDIAN_MEAL_RATING',]





# looping to make x-variables suitable for statsmodels
for val in x_variables:
    print(f"original_df_train['{val}'] +")


# This is my best model so far with an R-square value of 0.780.

# In[23]:


# merging X_train and y_train so that they can be used in statsmodels
original_df_train = pd.concat([X_train, y_train], axis = 1)


# Step 1: build a model
lm_5 = smf.ols(formula =  """REVENUE~ original_df_train['CROSS_SELL_SUCCESS'] +
original_df_train['TOTAL_MEALS_ORDERED'] +
original_df_train['CONTACTS_W_CUSTOMER_SERVICE'] +
original_df_train['LATE_DELIVERIES'] +
original_df_train['AVG_PREP_VID_TIME'] +
original_df_train['LARGEST_ORDER_SIZE'] +
original_df_train['MASTER_CLASSES_ATTENDED'] +
original_df_train['MEDIAN_MEAL_RATING'] +
original_df_train['TOTAL_PHOTOS_VIEWED'] +
original_df_train['out_AVG_PREP_VID_TIME'] +
original_df_train['out_MASTER_CLASSES_ATTENDED'] +
original_df_train['change_CONTACTS_W_CUSTOMER_SERVICE'] +
original_df_train['change_AVG_PREP_VID_TIME'] +
original_df_train['change_UNIQUE_MEALS_PURCH'] +
original_df_train['change_MEDIAN_MEAL_RATING']  """,
                                data = original_df_train)


# Step 2: fit the model based on the data
results = lm_5.fit()



# Step 3: analyze the summary output
print(results.summary())


# Instantiate, fit, predict, and score a linear regression model to train and test the data.

# In[24]:


# applying modelin scikit-learn

# preparing x-variables
original_df_data = original_df.loc[:,x_variables]


# preparing response variable
original_df_target = original_df.loc[:,'REVENUE']


# running train/test split again
X_train,X_test,y_train,y_test=train_test_split(
                           original_df_data,
                           original_df_target,
                           test_size = 0.25,
                           random_state = 222)


# In[25]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
                                        original_df_data,
                                        original_df_target,
                                        test_size = 0.25, 
                                        random_state = 222)

lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


# SCORING the results
print('Training Score:', lr.score(X_train, y_train).round(4))
print('Testing Score:',  lr.score(X_test, y_test).round(4))

# saving scoring data for future use
lr_train_score = lr.score(X_train, y_train).round(4)
lr_test_score  = lr.score(X_test, y_test).round(4)


# After plotting the Linerar regression model to check residuals ,it was clearly visible that there was no definte pattern to it .

# In my Analysis i used OLSR ,LASSO,RIDGE and after analyzing the results from all these models , OLSR gave me the best results .

# In[ ]:




