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
with open (file,'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
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

![Screenshot (120)](https://github.com/user-attachments/assets/d0e94cae-82ed-401d-95b5-8c44e95cc97f)

![Screenshot (121)](https://github.com/user-attachments/assets/7423b31a-2a28-443b-aeb3-04e1abf43dc5)

![Screenshot (122)](https://github.com/user-attachments/assets/ed9411b0-807d-41ef-b42a-398ab7b3356d)

![Screenshot (123)](https://github.com/user-attachments/assets/1341c856-9f20-4b6c-bee3-47498992500d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
