#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install xgboost


# In[6]:


import numpy as np
import pandas as pd
import xgboost as xg
import seaborn as sb 
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[7]:


df = pd.read_csv("car_data.csv")
df


# In[9]:


print("Check for any missing values: \n")
print(df.isnull().sum())


# In[11]:


print("Check statistical values: \n")
df.describe()


# In[13]:


print("The 'Car Name' column was removed, since in this case it does not effect the price prediction results")
df1= df.drop('Car_Name', axis=1)
df1


# In[15]:


df1['Current_Year'] = 2024
df1['No_of_Years'] = df1['Current_Year'] - df1['Year']
df1


# In[16]:


df1.drop((['Year','Current_Year']), axis=1, inplace=True)
df1


# In[17]:


#Turning categorical data into numerical data for Fuel Type, Seller Type, and Transmission
cat_num=pd.get_dummies(df1, drop_first=True)
cat_num


# In[20]:


#Finding the correlation between columns
cmap=cat_num.corr()
cmap


# In[22]:


c_feat = cmap.index 
plt.figure(figsize=(10,6))
g=sb.heatmap(cat_num[c_feat].corr(),annot=True,cmap="RdYlGn")


# # XG Boost Model 

# In[27]:


#Splitting the Data into Training Set
# Independent variables
#X = cd_con.drop("Selling_Price", axis = 1)

# Dependent variables
#Y = cd_con['Selling_Price']
X = cat_num.iloc[:, 2:9].values
Y = cat_num.iloc[:, 1].values


# In[28]:


X


# In[29]:


Y


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape
#Y_train will have same shape


# In[41]:


X_test.shape
#Y_test will have same shape


# In[42]:


#Train model with XGBoost

labelencoder_X_1 = LabelEncoder()
X[:, 3] = labelencoder_X_1.fit_transform(X[:, 4])

labelencoder_X_2 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])


labelencoder_X_3 = LabelEncoder()
X[:, 6] = labelencoder_X_3.fit_transform(X[:, 6])

#labelencoder_X_4 = LabelEncoder()
#X[:, 7] = labelencoder_X_4.fit_transform(X[:, 7])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X


# In[50]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder and transform y_train
y_train_encoded = label_encoder.fit_transform(y_train)

# Initialize and fit the classifier
classifier = xg.XGBClassifier()
classifier.fit(X_train, y_train_encoded)


# # Multi Linear Regression

# In[59]:


#TRANING MODEL
# Independent variables
X = cat_num.drop("Selling_Price", axis = 1)

# Dependent variables
y = cat_num['Selling_Price']


# In[60]:


X.head()


# In[62]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[63]:


from sklearn.linear_model import LinearRegression
# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)


# In[64]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test) # Make predictions on the testing data
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Coefficient of Determination (R^2 Score):', r2_score(y_test, y_pred))


# In[65]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[67]:


coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coefficients)
 #The coefficients for the 'Owner' and 'No_of_Years' features are negative (-0.903760 and -0.353801, respectively), 
 #suggesting that a higher number of owners and older cars tend to have lower selling prices


# In[ ]:




