# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NITHISH S
RegisterNumber: 212223220070 
*/
## Developed by: CHARUMATHI R
## RegisterNumber: 212222240021

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
## PLACEMENT DATA:
![267753589-cba641d7-4b64-474a-9df3-f8047b4ddc21](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/bb486926-9895-423d-a47b-db13cc50f3a1)

## Salary Data:
![267753686-b69592e3-fb46-446d-87a4-60e8dabf45a1](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/a886230c-b705-4c52-ba99-ea0900fbb5ec)

## Checking the null() function:
![267753782-196a08f0-0571-40f2-bfdf-b6e1d2b4fa8f](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/89f998fb-ea8d-4550-8c3b-a55f849bb7dc)
## Data Duplicate:

![267753891-3efb2a8c-6c60-4466-99b2-2c3c7b7a39b4](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/8115d2a7-da7b-460c-8776-df27d3c66c19)
## Print Data:
![267753963-37d05f23-2187-49d2-a871-7dbf5d7baca9](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/cb479a1e-f548-4df8-9fe3-e47627f81f08)

## Data-Status:
![267754049-d0b24ebb-4d7a-4956-b6e5-b87f65ccbeeb](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/a8aefc68-b1a8-4f44-b958-1bff4e7b64cb)
## Y_prediction array:
![267754328-81a5cd80-1fa0-48d8-a838-567b6e7a6676](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/2059a470-b1c5-43ce-abe7-49420a7e151a)
## Accuracy value:
![267754448-1ca21819-8baa-4312-aae8-1b094fe75ea6](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/7c61318a-8ea8-453e-b55a-a66a11892cf6)
## Confusion array:
![267754513-675efabe-006d-463a-b5f0-0cc4354ca37a](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/d4dfa415-32b4-4eb8-99a7-6e1245abb9a9)
## Classification Report:
![267754597-be3ab929-d71c-492a-8adc-9a054cf08983](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/953573ea-9306-45a3-ae50-267d64d37586)

## Prediction of LR:
![267754663-295b82c5-385c-4832-9d92-282a651946cb](https://github.com/Nithish23013509/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149038138/f40e3c8a-2705-4b59-8466-b14b489eea84)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
