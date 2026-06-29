import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error

#loading data
df=pd.read_csv("C:\\Users\\Lenovo\\Downloads\\house_price_50k_unclean.csv")

print(df['Location'].value_counts())
print(df.isnull().sum())

df['Size']=df['Size'].astype(str).str.lower().str.replace('sqft','').str.strip()
df['Size']=pd.to_numeric(df['Size'],errors='coerce')

#filling missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
num_cols=['Size','Bedrooms','Bathrooms','Year_Built']
df[num_cols]=imputer.fit_transform(df[num_cols])

#feature engineering
df['Age']=2026-df['Year_Built']
df=df.drop(columns=["Year_Built","Id"])
print(df.head())

#handling categorical columns
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['Location']=LE.fit_transform(df['Location'])
print(df.head())

X=df.drop(columns=['Price'])
y=df['Price']

#spliting the data into traning data and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#model building
from sklearn.ensemble  import RandomForestRegressor
Rfr=RandomForestRegressor(n_estimators=20,max_depth=8,random_state=42,n_jobs=-1)
Rfr.fit(X_train,y_train)
prediction=Rfr.predict(X_test)

#calculating r2_score,mean absolut error
print("R2_score",r2_score(prediction,y_test))
print("mean absolute error",mean_absolute_error(prediction,y_test))

#saving on uplaod on pickle
pickle.dump(Rfr,open("C:\\Users\\Lenovo\\Desktop\\langchain\\mlprojects\\house_price_prediction.pkl","wb"))
pickle.dump(LE,open("C:\\Users\\Lenovo\\Desktop\\langchain\\mlprojects\\labelencoder.pkl","wb"))
print(df['Location'].unique())


print(Rfr.get_params())
print(df.shape)





