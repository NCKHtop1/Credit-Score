#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("CreditScoreData.csv")
print(data.head())


# In[5]:


print(data.info())


# In[7]:


print(data.isnull().sum())


# In[9]:


data["Credit_Score"].value_counts()


# In[16]:


# Data Exploration
# Occupation of the person affects credit scores: 

fig = px.box(data,
             x="Occupation",
             color="Credit_Score",
             title="Credit Scores Based on Occupation",
             color_discrete_map= {'Poor':'red',
                                  'Standard':'yellow',
                                  'Good':'green'})
fig.show()


# In[19]:


# Annual_Income of the person affects credit scores: 
fig = px.box(data,
             x="Credit_Score",
             y="Annual_Income",
             color="Credit_Score",
             title="Credit Scores Based on Annual Income",
             color_discrete_map={'Poor':'red',
                                  'Standard':'Blue',
                                  'Good':'Yellow'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[21]:


# Number Bank Accounts
fig = px.box(data,
             x="Credit_Score",
             y="Num_Bank_Accounts",
             color="Credit_Score",
             title="Credit Score based on Number of Bank Accounts",
             color_discrete_map={'Poor':'red',
                                 'Standard':'blue',
                                 'Good':"purple"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[24]:


# Number Credit Card
fig = px.box(data,
             x="Credit_Score",
             y="Num_Credit_Card",
             color="Credit_Score",
             title="Credit Score based on Number of Credit Cards",
             color_discrete_map={'Poor':'Gray',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[27]:


# Interest rate
fig = px.box(data,
             x="Credit_Score",
             y="Interest_Rate",
             color="Credit_Score",
             title="Credit Score based on Interest Rate",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[33]:


# Number of Loans
fig = px.box(data,
             x="Credit_Score",
             y="Num_of_Loan",
             color="Credit_Score",
             title="Credit Scores based on Number of Loans taken by the person",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[35]:


# Delay form due date
fig = px.box(data,
             x="Credit_Score",
             y="Delay_from_due_date",
             color="Credit_Score",
             title="Credit Scores based on Average Number of Delay from due date",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[37]:


# Outstanding debt
fig = px.box(data,
             x="Credit_Score",
             y="Outstanding_Debt",
             color="Credit_Score",
             title="Credit Scores based on Outstanding debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[41]:


# Credit Utilization Ratio
fig = px.box(data,
             x="Credit_Score",
             y="Credit_Utilization_Ratio",
             color="Credit_Score",
             title="Credit Scores based on Credit Ultilization Ratio",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[43]:


# Credit History age
fig = px.box(data,
             x="Credit_Score",
             y="Credit_History_Age",
             color="Credit_Score",
             title="Credit Scores based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[44]:


# Total EMI per month
fig = px.box(data,
             x="Credit_Score",
             y="Total_EMI_per_month",
             color="Credit_Score",
             title="Credit Scores based on Total EMI per month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[45]:


# Amount invested monthly
fig = px.box(data,
             x="Credit_Score",
             y="Amount_invested_monthly",
             color="Credit_Score",
             title="Credit Scores based on Amount invested monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[49]:


# Month Balance
fig = px.box(data,
             x="Credit_Score",
             y="Monthly_Balance",
             color="Credit_Score",
             title="Credit Scores based on Monthly Balance left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'orange',
                                 'Good':"green"})
fig.update_traces(quartilemethod="exclusive")
fig.show()


# In[51]:


# CREDIT SCORE CLASSIFICATION MODEL
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard":1,
                                            "Good":2,
                                            "Bad":0})


# In[54]:


from sklearn.model_selection import train_test_split
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data[["Credit_Score"]])


# In[ ]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)


# In[ ]:


print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))
features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))

