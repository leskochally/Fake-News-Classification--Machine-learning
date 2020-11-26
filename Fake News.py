"""
MACHINE LEARNING PROJECT- FAKE NEWS DETECTION
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn

data= pd.read_csv("data.csv")


"checking NULL Values"
data.isnull().sum()
data= data.dropna(how='any',axis=0)

" CREATING TEST AND TRAIN DATASET"
x= data['Body']
y= data['Label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

"DataFlair - Initialize a TfidfVectorizer"
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

" DataFlair - Fit and transform train set, transform test set"
t_train=tfidf_vectorizer.fit_transform(x_train) 
t_test=tfidf_vectorizer.transform(x_test)



""" DIFFERENT MODELS"""


" PASSIVE AGRESSIVE CLASSIFIER"
from sklearn.linear_model import PassiveAggressiveClassifier

model=PassiveAggressiveClassifier(max_iter=50)
model.fit(t_train,y_train)
pred= model.predict(t_test)
accuracy= accuracy_score(y_test,pred)
print(round(accuracy*100,2))
confusionmat= confusion_matrix(y_test,pred,labels=[0,1])
print(confusionmat)
sn.heatmap(confusionmat,annot=True)


" KNeighborsClassifier"
from sklearn.neighbors import KNeighborsClassifier

model2= KNeighborsClassifier(n_neighbors=9)
model2.fit(t_train,y_train)
pred1= model2.predict(t_test)
accuracy1= accuracy_score(y_test,pred1)
print(round(accuracy1*100,2))
confusionmat1= confusion_matrix(y_test,pred1,labels=[0,1])
print(confusionmat1)
sn.heatmap(confusionmat1,annot=True)




"LOGISTIC REGRESSION"
from sklearn.linear_model import LogisticRegression

model3= LogisticRegression(solver='lbfgs')
model3.fit(t_train,y_train)
pred2= model3.predict(t_test)
accuracy2= accuracy_score(y_test,pred2)
print(round(accuracy2*100,2))
confusionmat2= confusion_matrix(y_test,pred2,labels=[0,1])
print(confusionmat2)
sn.heatmap(confusionmat2,annot=True)



"Support Vector Machine"
from sklearn.svm import SVC

model4 = SVC(kernel='rbf')
model4.fit(t_train, y_train)
pred3= model4.predict(t_test)
accuracy3= accuracy_score(y_test,pred3)
print(round(accuracy3*100,2))
confusionmat3= confusion_matrix(y_test,pred3,labels=[0,1])
print(confusionmat3)

"NAIVE CLASSIFIER"
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(t_train.todense(),y_train)
pred4= gnb.predict(t_test.todense())
accuracy4= accuracy_score(y_test,pred4)
print(round(accuracy4*100,2))
confusionmat4= confusion_matrix(y_test,pred4,labels=[0,1])
print(confusionmat4)

"PLOTTING OF ACCURACY"

s= [['NLP',98.75],['KNN',91.6],
        ['LOGISTIC REGRESSION',97],['SVM',56.39],['GAUSSIAN',89.6]]

df= pd.DataFrame(s,columns=['Algorithm','Accuracy'])
df.plot.barh(x='Algorithm', y='Accuracy')



