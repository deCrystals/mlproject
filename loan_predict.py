import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
%matplotlib inline
warnings.filterwarnings("ignore")
%cd "C:\Users\user\Desktop\portfolio"
train = pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")
train_original = train.copy()
test_original = test.copy()
train.info()
train.head()
train.describe()

train.shape

test.info()
test.shape
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize = True)
train['Loan_Status'].value_counts().plot.bar()
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
plt.show()

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24, 6), title='Dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16, 5))
plt.show()
train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle('')
              
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16, 5))
plt.show()
plt.figure(1)
plt.subplot(121)
sns.distplot(train['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16, 5))
plt.show()
              
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Dependent = pd.crosstab(train['Dependents'], train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Education=pd.crosstab(train['Education'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show()
Dependent.div(Dependent.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show
Credit_History= pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show()
Property_Area= pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis =0).plot(kind= 'bar', stacked=True, figsize=(4,4))
plt.show()

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
bins=[0, 1000, 3000, 42000]
group=['low', 'Average', 'High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'], bins, labels=group)
Coapplicant_Income_bin =pd.crosstab(train['Coapplicant_Income_bin'], train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked =True)
plt.xlabel('coapplicantIncome')
P=plt.ylabel('Percentage')
train['Total_Income'] =train['ApplicantIncome'] + train['CoapplicantIncome']
bins=[0, 2500, 4000, 6000, 8100]
group=['low', 'Average', 'High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'], bins, labels=group)
Total_Income_bin =pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked =True)
plt.xlabel('Total_Income')
P= plt.ylabel('Percentage')
bins= [0,100, 200, 700]
group=['low', 'Average', 'High']
train['LoanAmount_bin'] =pd.cut(train['LoanAmount'], bins, labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'], train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis =0).plot(kind ="bar", stacked=True)
plt.xlabel('Loan Amount')
P= plt.ylabel('Percentage')
train.head()
train = train.drop(columns =['Coapplicant_Income_bin','Total_Income', 'Total_Income_bin', 'LoanAmount_bin'])
train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace =True)
train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)
matrix = train.corr() 
f, ax=plt.subplots(figsize=(9,6))
sns.heatmap(matrix, vmax=.8, square=True, cmap='BuPu');
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Gender'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train.isnull().sum()
train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].value_counts()
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()
test.isnull().sum()
#filling null in test data
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)





#treating outliers
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])
train = train.drop('Loan_ID', axis =1)
train
test =test.drop('Loan_ID', axis =1)
test
train.info()
# separating target from train data
X = train.drop('Loan_Status', 1)
y = train.Loan_Status
y
X.info()
X= pd.get_dummies(X)
X.info()
X = X.drop('Self_Employed_Male', axis = 1)
X
train = pd.get_dummies(train)
train

train = train.drop('Self_Employed_Male',  axis =1)
train
test = pd.get_dummies(test)
test
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling =1,
                  max_iter=100, multi_class='obr', n_jobs=1, penalty='l2', random_state=1, solver='liblinear', tol=0.001, verbose=0, warm_start=False)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv, pred_cv)
pred_test =model.predict(test)
submission = pd.read_csv('sample_submission_49d68Cx.csv')
submission
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']
submission
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission


pd.DataFrame(submission, columns = ['Loan_ID', 'Loan_Status']).to_csv('loan_prediction.csv')