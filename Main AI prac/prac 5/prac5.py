import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,ShuffleSplit,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn import preprocessing 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,BaseEnsemble,GradientBoostingClassifier
from sklearn.svm import SVC,LinearSVC
import time
from matplotlib.colors import ListedColormap
from xgboost import XGBRegressor
from skompiler import skompile
from lightgbm import LGBMRegressor

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)

df=pd.read_csv("diabetes.csv")
print(df.head(),"\n")
print(df.shape,"\n")
print(df.describe(),"\n\n")
X=df.drop('Outcome',axis=1)
y=df['Outcome']

X_train=X.iloc[:600]
X_test=X.iloc[600:]
y_train=y[:600]
y_test=y[600:]
print("X_train Shape:",X_train.shape)
print("X_test Shape:",X_test.shape)
print("y_train Shape:",y_train.shape)
print("y_test Shape:",y_test.shape)

support_vector_classifier=SVC(kernel="linear").fit(X_train,y_train)
print("\n\n",support_vector_classifier)
print("\n",support_vector_classifier.C)
print("\n",support_vector_classifier)

y_pred=support_vector_classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print("\n",cm,"\n")
print("Our Accuracy is: ",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]),"\n")
print("Accuracy score: ",accuracy_score(y_test,y_pred),"\n")
print("Classification Report : \n",classification_report(y_test,y_pred),"\n")

#K-Fold Cross Validation
print(support_vector_classifier,"\n")
accuracies= cross_val_score(estimator=support_vector_classifier,X=X_train,y=y_train,cv=10)
print("Average Accuracy: {:.2f}%".format(accuracies.mean()*100),"\n")
print("Standard Deviation of Accuracies: {:.2f}%".format(accuracies.std()*100),"\n")

print(support_vector_classifier.predict(X_test)[:10],"\n")

svm_params = {"C":np.arange(1,20)}

svm= SVC(kernel="linear")
svm_cv=GridSearchCV(svm,svm_params,cv=8)

start_time=time.time()
svm_cv.fit(X_train,y_train)
elapsed_time=time.time()-start_time
print(f"Elapsed time for Support Vector Regression cross validation: "f"{elapsed_time:.3f} seconds\n")

print("Best Score : ",svm_cv.best_score_,"\n")

print("Best Parameter: ",svm_cv.best_params_,"\n")

svm_tuned=SVC(kernel="linear",C=2).fit(X_train,y_train)
print(svm_tuned,"\n")

y_pred=svm_tuned.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm,"\n")

print("Our Accuracy is: ",(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]),"\n")
print("Accuracy score: ",accuracy_score(y_test,y_pred),"\n")
print("Classification Report : \n",classification_report(y_test,y_pred),"\n")
