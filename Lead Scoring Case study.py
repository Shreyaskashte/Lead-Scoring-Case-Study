#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing important libraries.
import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


# Importing all datasets
lead = pd.read_csv('Leads.csv')


# ## Inspecting the Dataframe

# In[3]:


# Let's see the head of our dataset
lead.head()


# In[4]:


# check the size of dataset.
lead.shape


# In[5]:


#Check the data types of dataset
lead.info()


# In[ ]:





# ### Data preparation

# #### Converting some binary variables (Yes/No) to 0/1

# In[6]:


varlist = ['Do Not Email', 'Do Not Call', 'Search', 'Magazine', 'Newspaper Article','X Education Forums','Newspaper','Digital Advertisement'
           ,'Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content',
           'I agree to pay the amount through cheque','A free copy of Mastering The Interview']

#defining the map function
def binary_map(x):
    return x.map({'Yes':1,'No':0})

#Applying functions on variables
lead[varlist] = lead[varlist].apply(binary_map)


# In[7]:


#Let's check the dataset again
lead.head()


# In[8]:


lead.info()


# In[ ]:





# In[9]:


#Lets check if there is null value or not.
lead.isnull().sum()


# In[10]:


#Let's check null value in percentage.
round((((lead.isnull().sum())/len(lead))*100),2)


# As we can see there is null value present in dataset. So,we need to drop some column of null values. Let's set the 35% cut-off
# for null values.We will drop the column which have 35% above null values.

# In[11]:


lead = lead.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score',
                 'Tags','Lead Quality'],1)


# In[12]:


round((((lead.isnull().sum())/len(lead))*100),2)


# For other columns,which have null values less than 35%, we will impute them.

# In[13]:


# Check for unique values in dataset
lead.nunique()


# We need to drop 'Prospect ID' and 'Lead Number'. Also,let's drop 'Last Notable Activity', it is same as 'Last Activity' and also it didn't give much of information.

# In[14]:


lead = lead.drop(['Prospect ID','Lead Number','Last Notable Activity'],1)


# In[15]:


lead.columns


# In[ ]:





# In[ ]:





# ## Checking Balance In Dataset Columns

# In this we will check the categorical columns with distribution of data in column and we will remove the most imbalance columns.

# #### Lead Origin

# In[16]:


lead['Lead Origin'].value_counts()


# It is balance between values of dataset.

# #### Lead Source

# In[17]:


lead['Lead Source'].value_counts()


# We need to organise the other categorise which have less value counts into one category.

# In[18]:


lead['Lead Source'] = lead['Lead Source'].replace(['Facebook','bing','google','Click2call','Press_Release','Social Media','Live Chat',
                                   'youtubechannel','testone','Pay per Click Ads','welearnblog_Home',
                                   'WeLearn','blog','NC_EDM'], 'Others')


# In[19]:


lead['Lead Source'].value_counts()


# It is also have balance dataset values.

# #### Do Not Email

# In[20]:


lead['Do Not Email'].value_counts()


# This variable we need to drop due to its imbalance data distribution.

# #### Do Not Call

# In[21]:


lead['Do Not Call'].value_counts()


# This variable we need to drop due to its imbalance data distribution.

# #### Converted

# In[22]:


lead['Converted'].value_counts()


# #### Last Activity

# In[23]:


lead['Last Activity'].value_counts()


# It is also balance variable for value in dataset.

# #### Country

# In[24]:


(lead['Country'].value_counts()/ len(lead['Country']))*100


# By looking at country variable we can say it is mainly focused on country 'India'.We  can  not  get the  information other than country 'India'.The most values are saturated in 'India'.

# ####  Specialization
# 

# In[25]:


lead['Specialization'].value_counts()


# If we ignore 'Select' category, the data is distributed in all category of specialization is in balance.

# #### How did you hear about X Education

# In[26]:


lead['How did you hear about X Education'].value_counts()


# In[27]:


lead['How did you hear about X Education'].isnull().sum()


# In this column it is mostly saturated in 'Select' and in null values. So, we can drop this column because 'select' category is as good as NaN.

# #### What is your current occupation

# In[28]:


lead['What is your current occupation'].value_counts()


# We can use this occupation column.

# #### What matters most to you in choosing a course

# In[29]:


lead['What matters most to you in choosing a course'].value_counts()


# We can drop this column due to it is saturated only in one category

# #### Search

# In[30]:


lead['Search'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Magazine

# In[31]:


lead['Magazine'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Newspaper Article

# In[32]:


lead['Newspaper Article'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### X Education Forums

# In[33]:


lead['X Education Forums'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Digital Advertisement

# In[34]:


lead['Digital Advertisement'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Through Recommendations

# In[35]:


lead['Through Recommendations'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Receive More Updates About Our Courses

# In[36]:


lead['Receive More Updates About Our Courses'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Update me on Supply Chain Content

# In[37]:


lead['Update me on Supply Chain Content'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Get updates on DM Content

# In[38]:


lead['Get updates on DM Content'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### Lead Profile

# In[39]:


lead['Lead Profile'].value_counts()


# #### City

# In[40]:


lead['City'].value_counts()


# We need to drop the 'City'  column because it does not provie information except 'Mumbai'  and  other values  are saturated  in  'Select'  category.

# #### I agree to pay the amount through cheque

# In[41]:


lead['I agree to pay the amount through cheque'].value_counts()


# We need to drop this column, due to imbalance in data distribution.

# #### A free copy of Mastering The Interview

# In[42]:


lead['A free copy of Mastering The Interview'].value_counts()


# It is well balanced variable in his category,so we can use this variable.

# In[43]:


#Let's drop the imbalance variables.
lead = lead.drop(['Do Not Call','How did you hear about X Education','What matters most to you in choosing a course',
'Search','Magazine','Newspaper','Newspaper Article','X Education Forums','Digital Advertisement','Through Recommendations',
'Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content',
'I agree to pay the amount through cheque','City','Country'],1)




# # 

# # Imputing Null Values 

# In  this  section  we  will  fill  the  null  values  with  'NaN'  and  also  replace  the  'City' , 'lead profile' ,   'Specialization'  columns  'Select'  values.
# 

# In[44]:


(lead.isnull().sum()/len(lead))*100


# We will drop the null values, which is less than 2% in continuous columns in 'Lead Source',  'TotalVisits' and  'Page Views Per Visit'.

# In[45]:


lead.dropna(subset=['Lead Source','TotalVisits','Page Views Per Visit','Last Activity'],axis=0,inplace=True)


# #### Let's Impute Null Values

# #### Specialization
# We need to fill the 'Select' and null values because both are the same in 'Specialization'.

# In[46]:


lead['Specialization'].replace('Select','No Specialization selection', inplace=True)


# In[47]:


lead["Specialization"].fillna("No Specialization selection", inplace = True)


# In[48]:


lead["Specialization"].value_counts()


# #### What is your current occupation
# Let's fill the null values of "What is your current occupation" with "No occupation selection".

# In[49]:


lead["What is your current occupation"].value_counts()


# In[50]:


lead["What is your current occupation"].fillna("Unemployed", inplace = True)


# In[51]:


lead["What is your current occupation"].value_counts()


# #### Lead Profile

# In[52]:


lead["Lead Profile"].value_counts()


# In[53]:


lead['Lead Profile'].replace('Select','No lead profile', inplace=True)


# In[54]:


lead["Lead Profile"].fillna("No lead profile", inplace = True)


# In[55]:


lead["Lead Profile"].value_counts()


# #### Let's check for column null values again

# In[56]:


(lead.isnull().sum()/len(lead))*100


# Ther is no null values present in our data set now . We can move to the EDA analysis.

# In[ ]:





# # 

# ## Exploratory Data Analysis

# ### Bivariate Analysis

# ##### Lead Origin vs Converted

# In[57]:


plt.figure(figsize=[10,5])
sns.countplot(lead['Lead Origin'],hue=lead['Converted'])
plt.xticks(rotation=0)
plt.show()


# We can see that there is 'Landing Page Submission' have highest number of  converted customer  and  'not converted'  customer,but it is 'Lead Add Form' that have very good ratio of customer  'conversion' than 'non conversion' into course.

# ##### Lead source vs Converted

# In[58]:


plt.figure(figsize=[15,5])
sns.countplot(lead['Lead Source'],hue=lead['Converted'])
plt.xticks(rotation=0)
plt.show()


# In  above  we  can  see  'google' and 'direct traffic'  has most converted values.
# But 'coversion' rate is high in 'Reference' and 'Welingak website' than  'non converion' rate.

# ##### 'Do not email' vs Converted

# In[59]:


plt.figure(figsize=[10,5])
sns.countplot(lead['Do Not Email'],hue=lead['Converted'])
plt.xticks(rotation=0)
plt.show()


# 'Do not email'  has  no  effect  in  'conversion'  of  customer  in  course.

# ##### 'Last Activity' vs Converted

# In[60]:


plt.figure(figsize=[15,5])
sns.countplot(lead['Last Activity'],hue=lead['Converted'])
plt.xticks(rotation=90)
plt.show()


# 'Email opened' and 'SMS sent' has high value of conversion.But 'SMS sent'  and  'Had a Phone Conversation'  has high positive rate of conversion.

# ##### Specialization vs Converted

# In[61]:


plt.figure(figsize=[15,5])
sns.countplot(lead['Specialization'],hue=lead['Converted'])
plt.xticks(rotation=90)
plt.show()


# In  specialization  'Finance Management' , 'Human Resource Management' , 'Marketing Management'  and  'Operations Management'
# has high rate of conversions in specialization.

# ##### 'What is your current occupation' vs Converted

# In[62]:


plt.figure(figsize=[10,5])
sns.countplot(lead['What is your current occupation'],hue=lead['Converted'])
plt.xticks(rotation=30)
plt.show()


# In 'Unemployed'  has high rate of  in  'conversion' and in 'no conversion' ,  but in  'working professional' there 'conversion' rate is higher than 'no conversion' .

# ##### 'Lead Profile' vs Converted

# In[63]:


plt.figure(figsize=[10,5])
sns.countplot(lead['Lead Profile'],hue=lead['Converted'])
plt.xticks(rotation=40)
plt.show()


# In 'lead profile'  we  have high 'conversion rate' than any other category.

# ##### 'A free copy of Mastering The Interview' vs Converted

# In[64]:


plt.figure(figsize=[10,5])
sns.countplot(lead['A free copy of Mastering The Interview'],hue=lead['Converted'])
plt.xticks(rotation=0)
plt.show()


# 'A free copy of mastering the interview' does  not  have  any effect on 'conversion'.

# ### Univariate Analysis

# ##### TotalVisits

# In[65]:


plt.figure(figsize=[15,5])
sns.histplot(x='TotalVisits', data=lead, kde=True)


# ##### Total Time Spent on Website

# In[66]:


plt.figure(figsize=[15,5])
sns.histplot(x='Total Time Spent on Website', data=lead, kde=True)


# ##### Page Views Per Visit

# In[67]:


plt.figure(figsize=[15,5])
sns.histplot(x='Page Views Per Visit', data=lead, kde=True)


# In above 'univariate analysis'  we  can  see  in  all  variables  there  is  presence  of  outliers. So,  we  need  to  treat  the outliers.

# # 

# ## Outlier's Treatment
# We  need  to  treat  outliers  in  continuous  variable.

# #### Total visit

# In[68]:


# Visualising the distribution of 'TotalVisits':
lead['TotalVisits'].value_counts().plot.box()
plt.show()


# In[69]:


# Checking the outliers in percentages.
lead['TotalVisits'].describe()


# In[70]:


# Capping the variable for outliers.
Q1 = lead['TotalVisits'].quantile(0.01)
Q3 = lead['TotalVisits'].quantile(0.85)
IQR = Q3-Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
lead_1 = lead.loc[(lead['TotalVisits']>lower_limit) & (lead['TotalVisits']<upper_limit)]


# In[71]:


#Check the variable after capping the variable.
sns.boxplot(lead_1['TotalVisits'])


# #### Total Time Spent on Website

# In[72]:


# Visualising the distribution of 'Total Time Spent on Website':
lead['Total Time Spent on Website'].value_counts().plot.box()
plt.show()


# In[73]:


# Checking the outliers in percentages.
lead['Total Time Spent on Website'].describe()


# In[74]:


# Capping the variable for outliers.
Q1 = lead['Total Time Spent on Website'].quantile(0.01)
Q3 = lead['Total Time Spent on Website'].quantile(0.99)
IQR = Q3-Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
lead_1 = lead.loc[(lead['Total Time Spent on Website']>lower_limit) & (lead['Total Time Spent on Website']<upper_limit)]


# In[75]:


#Check the variable after capping the variable.
sns.boxplot(lead_1['Total Time Spent on Website'])


# In[ ]:





# #### Page Views Per Visit

# In[76]:


# Visualising the distribution of 'Page Views Per Visit':
lead['Page Views Per Visit'].value_counts().plot.box()
plt.show()


# In[77]:


# Checking the outliers in percentages.
lead['Page Views Per Visit'].describe()


# In[78]:


# Capping the variable for outliers.
Q1 = lead['Page Views Per Visit'].quantile(0.01)
Q3 = lead['Page Views Per Visit'].quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1 - (1.5*IQR)
upper_limit = Q3 + (1.5*IQR)
lead_1 = lead.loc[(lead['Page Views Per Visit']>lower_limit) & (lead['Page Views Per Visit']<upper_limit)]


# In[79]:


#Check the variable after capping the variable.
sns.boxplot(lead_1['Page Views Per Visit'])


# In[ ]:





# In[ ]:





# In[80]:


# Check correlation between continuous variables.
Cont_column = lead.drop(['Do Not Email','A free copy of Mastering The Interview'],1)
sns.heatmap(Cont_column.corr(), cmap="coolwarm", annot=True)
plt.show()


# ## Building the model

# In[81]:


# Check the columns before building dummy variable.
lead_1.columns


# In[ ]:





# #### Let's build the dummy variables before splitting

# In[82]:


dummy = pd.get_dummies(lead_1,columns=['Lead Origin','Lead Source','Do Not Email','Last Activity','Specialization',
                                     'What is your current occupation','Lead Profile',
                                       'A free copy of Mastering The Interview'],drop_first=True)
dummy


# In[83]:


# Putting feature variable to X
X = dummy.drop(['Converted'], axis=1)

X.head()


# In[84]:


# Putting converted variable to y
y = dummy['Converted']

y.head()


# In[85]:


# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[86]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[87]:


# Let's see the correlation matrix
plt.figure(figsize=[100,100])
sns.heatmap(dummy.corr(), cmap="coolwarm", annot=True)
plt.show()


# #### Dropping highly correlated dummy variables

# In[88]:


X_test = X_test.drop (['What is your current occupation_Unemployed','Lead Profile_No lead profile','Specialization_No Specialization selection'], 1)
X_train= X_train.drop(['What is your current occupation_Unemployed','Lead Profile_No lead profile','Specialization_No Specialization selection'], 1)


# #### Checking the Correlation Matrix

# After dropping highly correlated variables now let's check the correlation matrix again.

# In[89]:


plt.figure(figsize=[100,100])
sns.heatmap(X_test.corr(), cmap="coolwarm", annot=True)
plt.show()


# In[90]:


import statsmodels.api as sm


# In[91]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ## Feature Selection Using RFE

# In[92]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[93]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train,y_train)


# In[94]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[95]:


col = X_train.columns[rfe.support_]
col


# ##### Assessing the model with StatsModels

# #### Iteration 1

# In[96]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[97]:


# Dropping the variable which have p-value > 0.05 .
col = col.drop(['What is your current occupation_Housewife'],1)


# #### Iteration 2

# In[98]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:





# #### Iteration 3

# In[99]:


# Dropping the variable which have p-value > 0.05 .
col = col.drop(['Lead Profile_Lateral Student'],1)


# In[100]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
print(res.summary())


# #### Check the VIF

# In[101]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ###### All P-values and VIF is in limit so, we will make prediction on train set

# # 

# ### Prediction on train data set

# In[102]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred


# ##### Creating a dataframe with the actual converted and the predicted probabilitie

# In[103]:


y_train_pred_final = pd.DataFrame({'converted':y_train.values , 'convert_prob':y_train_pred})
y_train_pred_final['Lead Number'] = y_train.index
y_train_pred_final.head()


# ##### Creating new column 'predicted' with 1 if convert_prob > 0.5 else 0

# In[104]:


y_train_pred_final['predicted'] = y_train_pred_final.convert_prob.map(lambda x : 1 if x >0.5 else 0)
y_train_pred_final.head()


# In[105]:


# Confusion matrix 
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.converted, y_train_pred_final.predicted )
print(confusion)


# In[106]:


# Predicted     not_convert    convert
# Actual
# not_convert        3393       429
# convert             660      1723 


# In[107]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.converted, y_train_pred_final.predicted))


# In[108]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[109]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[110]:


# Let us calculate specificity
TN / float(TN+FP)


# In[111]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[112]:


# positive predictive value 
print (TP / float(TP+FP))


# In[113]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Plotting the ROC Curve

# In[114]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[115]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.converted, y_train_pred_final.convert_prob, drop_intermediate = False )


# In[116]:


draw_roc(y_train_pred_final.converted, y_train_pred_final.convert_prob)


# In[117]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.convert_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[118]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[119]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[120]:


y_train_pred_final['final_predicted'] = y_train_pred_final.convert_prob.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()


# In[121]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.converted, y_train_pred_final.final_predicted)


# In[122]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.converted, y_train_pred_final.final_predicted )
confusion2


# In[123]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[124]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[125]:


# Let us calculate specificity
TN / float(TN+FP)


# In[126]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[127]:


# positive predictive value 
print (TP / float(TP+FP))


# In[128]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Precision and Recall

# In[129]:


#Precision
confusion2[1,1]/(confusion2[0,1]+confusion2[1,1])


# In[130]:


#Recall
confusion2[1,1]/(confusion2[1,0]+confusion2[1,1])


# In[ ]:





# ### Precision and recall tradeoff

# In[131]:


from sklearn.metrics import precision_recall_curve
y_train_pred_final.converted, y_train_pred_final.predicted


# In[132]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.converted, y_train_pred_final.convert_prob)


# In[133]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.converted, y_train_pred_final.convert_prob)
p, r, thresholds = precision_recall_curve(y_train_pred_final.converted, y_train_pred_final.convert_prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[ ]:





# # 

# ### Making prediction on test set 

# In[134]:



X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_test.head()


# In[135]:


X_test = X_test[col]
X_test.head()


# In[136]:


X_test_sm = sm.add_constant(X_test)


# In[137]:


y_test_pred = res.predict(X_test_sm)


# In[138]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[139]:


# Let's see the head
y_pred_1.head()


# In[140]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[141]:


# Putting CustID to index
y_test_df['Lead Number'] = y_test_df.index


# In[142]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[143]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[144]:


y_pred_final.head()


# In[145]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})


# In[146]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[147]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.35 else 0)


# In[148]:


y_pred_final.head()


# In[149]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)


# In[150]:


confusion3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion3


# In[151]:


TP1 = confusion3[1,1] # true positive 
TN1 = confusion3[0,0] # true negatives
FP1 = confusion3[0,1] # false positives
FN1 = confusion3[1,0] # false negatives


# In[152]:


# Let's see the sensitivity of our logistic regression model
TP1 / float(TP1+FN1)


# In[153]:


# Let us calculate specificity
TN1 / float(TN1+FP1)


# In[154]:


#Precision
confusion3[1,1]/(confusion3[0,1]+confusion3[1,1])


# In[155]:


#Recall
confusion3[1,1]/(confusion3[1,0]+confusion3[1,1])


# In[ ]:





# ### Calculating F1 score:

# In[156]:


# F1 score on training set:
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_train = 2 * (precision * recall) / (precision + recall)
F1_train


# In[157]:


# F1 score on test set:
precision_t = TP1 / (TP1 + FP1)
recall_t = TP1 / (TP1 + FN1)
F1_test = 2 * (precision_t * recall_t) / (precision_t + recall_t)
F1_test


# In[ ]:





# In[ ]:





# # Inferences:

# - The Score on train and test set were:
#       on training set:
#         1. accuracy: 81.32%
#         2. sensitivity: 80.78%
#         3. specificity: 81.65%
#         4. precision: 73.30%
#         5. recall: 80.78%
#         6. F1 score: 0.76
#       on test set:
#         1. accuracy: 82.25%
#         2. sensitivity: 82.18%
#         3. specificity: 82.29%
#         4. precision: 72.24%
#         5. recall: 82.18%
#         6. F1 score: 0.77
# 

# ### Conclusion : 

# #### Conclusion on Model
# ##### Positive correlations
# From 'Lead Origin' we have 'lead add form' high positive correalation coefficient 3.187 with conversion of customer.
# 
# In 'Lead Source' there is 'Welingak Websiite' has positive correalation coefficient 2.7448.
# 
# In 'What is your current occupation' the 'working professional' has positive correalation coefficient 2.6399 . 
# 
# In above positive relations customer converion probability in course is very high.
# ##### Negative correlations
# From 'Lead Profile' we have 'student from some school' high negative correalation coefficient -1.7992 with conversion of customer.
# 
# Those who have select 'Yes' in 'Do not email'  has negative correlation -1.5516.
# 
# Those whose "Last activity' was 'Olark Chat Conversion' has negative correlation -1.4766.
# 
# In above negative relations customer converion probability in course is very low.
# 
# 
# #### Conclusion on Inferences
# As we can see we got the accuracy,sensitivity and specificity are nearly equal to each other in both set around 80%.
# 
# Precision is the correctly predicted positive cases by the classifier. In test it is 73.30% and in test dataset 72.24%.
# 
# The recall is calculated by taking the proportion of correctly identified positive inputs.In test it is 80.78% and in test dataset 82.18%.
# 
# F1 score or F measure is also a measure of the test’s accuracy. It is defined as a weighted mean of precision and recall. It has its maximum value at 1 and worst at 0.In test it is 0.76 and in test dataset 0.77.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




