# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
# CODE
```python
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df
df.isnull().sum()
df.info()
sns.heatmap(df.isnull())
df["Age"]=df["Age"].fillna(df["Age"].dropna().median())
df.loc[df['Embarked']=='S','Embarked']=0
df.loc[df['Embarked']=='C','Embarked']=1
df.loc[df['Embarked']=='Q','Embarked']=2
df['Embarked']
drop_elementindataset=['Cabin','Ticket','Name']
df=df.drop(drop_elementindataset,axis=1)
df['Embarked']=df['Embarked'].fillna(df['Embarked'].median())
df.loc[df['Sex']=='male','Sex']=0
df.loc[df['Sex']=='female','Sex']=1
df['Sex']
import seaborn as sns
import matplotlib.pyplot as plt
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
sns.heatmap(df.isnull(),cbar=False)
df.Survived.value_counts(normalize=True).plot(kind='bar')
plt.show()
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
fig = plt.figure(figsize=(18,6))
df.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = df.drop("Survived",axis=1)
y = df["Survived"]
mdlsel = SelectKBest(chi2,k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(11)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
target = df['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = df[data_features_names].values
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')
my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)
print("Random forest score: ",accuracy_score(y_test,target_predict))
from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUPUT
![Screenshot 2023-05-24 083753](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/fa8b3b5c-e399-4aa8-a028-97db7a0dd059)
![Screenshot 2023-05-24 083800](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/6453fe27-7375-4602-9a5e-3f8ef0b15a26)
![Screenshot 2023-05-24 083820](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/5987d9a0-8798-41eb-bb58-3b5043575c18)
![Screenshot 2023-05-24 083833](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/c475aa54-0081-4e36-bcf7-85580f11cef4)
![Screenshot 2023-05-24 083844](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/8ef6bed3-ff53-46ad-b278-a5f86b191281)
![Screenshot 2023-05-24 083923](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/26457838-69ed-41c4-8048-13a0bc7d934c)
![Screenshot 2023-05-24 083943](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/0c1e225a-6a01-4640-ae55-a36b587fb91c)
![Screenshot 2023-05-24 083951](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/2a280694-75f2-409f-9bad-eba70c8516b7)
![Screenshot 2023-05-24 084010](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/79febd2c-f8bf-4762-ad54-e62f88939589)
![Screenshot 2023-05-24 084037](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/10f21b2a-63de-46a7-8b19-3ebfccde9ac7)
![Screenshot 2023-05-24 084047](https://github.com/BaskaranV15/Ex-07-Feature-Selection/assets/118703522/6ba4190a-f839-4a0e-9405-7c6938dec6f5)















