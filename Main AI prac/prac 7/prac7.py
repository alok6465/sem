from warnings import filterwarnings
filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns

df=pd.read_csv('Book1.csv')

print(df.head(11))
print(df.tail())
print(df.info())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Sore Throat']=le.fit_transform(df['Sore Throat'])
df['Fever']=le.fit_transform(df['Fever'])
df['Swollen Glands']=le.fit_transform(df['Swollen Glands'])
df['Congestion']=le.fit_transform(df['Congestion'])
df['Headache']=le.fit_transform(df['Headache'])
df['Diagnosis']=le.fit_transform(df['Diagnosis'])

print(df.info())
print(df.head(11))

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Sore Throat'],data=df)
plt.title("Category wise count of Sore Throat")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Fever'],data=df)
plt.title("Category wise count of Fver")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Swollen Glands'],data=df)
plt.title("Category wise count of Swallen Glands")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Congestion'],data=df)
plt.title("Category wise count of Congestion")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Headache'],data=df)
plt.title("Category wise count of Headache")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

fig,ax=plt.subplots(figsize=(6,6))
sns.countplot(x=df['Diagnosis'],data=df)
plt.title("Category wise count of Diagnosis")
plt.xlabel("category")
plt.ylabel("Count")
plt.show()

X=df.drop('Diagnosis',axis=1)
y=df['Diagnosis']

classifier=MultinomialNB()
print(classifier.fit(X,y))

classifier=CategoricalNB()
print(classifier.fit(X,y))

classifier=GaussianNB()
print(classifier.fit(X,y))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

classifier=MultinomialNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("Confusion matrix :\n ",confusion_matrix(y_test,y_pred))
print("accuracy_score : ",accuracy_score(y_test,y_pred))
print("precision_score : ",precision_score(y_test,y_pred))
print("recall_score : ",recall_score(y_test,y_pred))
print("f1_score : ",f1_score(y_test,y_pred))
print("classification_report : \n",classification_report(y_test,y_pred))



