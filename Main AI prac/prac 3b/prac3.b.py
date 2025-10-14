import numpy as np
import pandas as pd
from sklearn import tree 

Play_Tennis= pd.read_csv("PlayTennis.csv")
print(Play_Tennis,"\n\n")

from sklearn.preprocessing import LabelEncoder
Le= LabelEncoder()

Play_Tennis['Outlook']=Le.fit_transform(Play_Tennis['Outlook'])
Play_Tennis['Temperature']=Le.fit_transform(Play_Tennis['Temperature']) 
Play_Tennis['Humidity']=Le.fit_transform(Play_Tennis['Humidity']) 
Play_Tennis['Wind']=Le.fit_transform(Play_Tennis['Wind']) 
Play_Tennis['Play Tennis']=Le.fit_transform(Play_Tennis['Play Tennis']) 
print(Play_Tennis,"\n\n")

y=Play_Tennis['Play Tennis']
X=Play_Tennis.drop(['Play Tennis'], axis=1) 

clf=tree.DecisionTreeClassifier(criterion='entropy') 
clf=clf.fit(X, y)
print(tree.plot_tree(clf),"\n\n") 
print("Decision Tree Text Representation:\n", tree.export_text(clf, feature_names=list(X.columns)))

import graphviz
dot_data=tree.export_graphviz(clf, out_file=None) 
graph=graphviz.Source(dot_data)
print(graph)
 # python shell can't print graph image so, it will print the code for diagraph implementation

X_pred=clf.predict(X)
print(X_pred==y,"\n")

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))  
tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (Matplotlib)")
plt.show()