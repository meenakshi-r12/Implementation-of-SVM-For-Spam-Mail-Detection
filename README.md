# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries.
2. Use chardet to detect the encoding of the file spam.csv.
3. Load the CSV file using the encoding (in this case, 'windows-1252').
4. Display the first few rows and general information.
5. Ensure the dataset does not contain null or missing values.
6. Feature (x) = text messages (column "v2").
7. Label (y) = spam/ham classification (column "v1").
8. Split the Data into Training and Testing Sets.
9. Use CountVectorizer to convert text data into numerical vectors.
10. Use Support Vector Classifier to train on the vectorized data.
11. Predict whether messages are spam or not.
12. Use accuracy score to evaluate the model performance.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MEENAKSHI.R
RegisterNumber:  212224220062
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

![WhatsApp Image 2025-05-25 at 09 01 12_31b94e50](https://github.com/user-attachments/assets/3daf7528-bb9d-4459-938a-7bbb321cd3c8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
