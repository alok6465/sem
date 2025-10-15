dict_hn={'Arad':336,'Bucharest':0,'Craiova':160,'Drobeta':242,'Eforie':161,
         'Fagaras':176,'Giurgiu':77,'Hirsova':151,'Iasi':226,'Lugoj':244,
         'Mehadia':241,'Neamt':234,'Oradea':380,'Pitesti':100,'Rimnicu':193,
         'Sibiu':253,'Timisoara':329,'Urziceni':80,'Vaslui':199,'Zerind':374}

dict_gn=dict(
Arad=dict(Zerind=75,Timisoara=118,Sibiu=140),
Bucharest=dict(Urziceni=85,Giurgiu=90,Pitesti=101,Fagaras=211),
Craiova=dict(Drobeta=120,Pitesti=138,Rimnicu=146),
Drobeta=dict(Mehadia=75,Craiova=120),
Eforie=dict(Hirsova=86),
Fagaras=dict(Sibiu=99,Bucharest=211),
Giurgiu=dict(Bucharest=90),
Hirsova=dict(Eforie=86,Urziceni=98),
Iasi=dict(Neamt=87,Vaslui=92),
Lugoj=dict(Mehadia=70,Timisoara=111),
Mehadia=dict(Lugoj=70,Drobeta=75),
Neamt=dict(Iasi=87),
Oradea=dict(Zerind=71,Sibiu=151),
Pitesti=dict(Rimnicu=97,Bucharest=101,Craiova=138),
Rimnicu=dict(Sibiu=80,Pitesti=97,Craiova=146),
Sibiu=dict(Rimnicu=80,Fagaras=99,Arad=140,Oradea=151),
Timisoara=dict(Lugoj=111,Arad=118),
Urziceni=dict(Bucharest=85,Hirsova=98,Vaslui=142),
Vaslui=dict(Iasi=92,Urziceni=142),
Zerind=dict(Oradea=71,Arad=75)
)
























































































































































































































































































































































































































































































''' Prac 1.a BFS
import queue as Q                  
from RMP import dict_gn

start='Arad'
goal='Bucharest'
result=''

def BFS(city, cityq, visitedq):
    global result
    if city==start:
        result=result+' '+city
    for eachcity in dict_gn[city].keys():
        if eachcity==goal:
            result=result+' '+eachcity
            return
        if eachcity not in cityq.queue and eachcity not in visitedq.queue:
            cityq.put(eachcity)
            result=result+' '+eachcity
    visitedq.put(city)
    BFS(cityq.get(),cityq,visitedq)

def main():
    cityq=Q.Queue()
    visitedq=Q.Queue()
    BFS(start, cityq, visitedq)
    print("BFS Traversal from ",start," to ",goal," is: ")
    print(result)
    
main()



prac 21.b IDDFS
import queue as Q
from RMP import dict_gn

start='Arad'
goal='Bucharest'
result=''

def DLS(city, visitedstack, startlimit, endlimit):
    global result
    found=0
    result=result+city+' '
    visitedstack.append(city)
    if city==goal:
        return 1
    if startlimit==endlimit:
        return 0
    for eachcity in dict_gn[city].keys():
        if eachcity not in visitedstack:
            found=DLS(eachcity, visitedstack, startlimit+1, endlimit)
            if found:
                return found

def IDDFS(city, visitedstack, endlimit):
    global result
    for i in range(0, endlimit):
        print("Searching at Limit: ",i)
        found=DLS(city, visitedstack, 0, i)
        if found:
            print("Found")
            break
        else:
            print("Not Found! ")
            print(result)
            print("-----")
            result=' '
            visitedstack=[]

def main():
    visitedstack=[]
    IDDFS(start, visitedstack, 9)
    print("IDDFS Traversal from ",start," to ", goal," is: ")
    print(result)


main()       


Prac 2.a A*
import queue as Q
from RMP import dict_gn
from RMP import dict_hn

start='Arad'
goal='Bucharest'
result=''

def get_fn(citystr):
    cities=citystr.split(" , ")
    hn=gn=0
    for ctr in range(0, len(cities)-1):
        gn=gn+dict_gn[cities[ctr]][cities[ctr+1]]
    hn=dict_hn[cities[len(cities)-1]]
    return(hn+gn)

def expand(cityq):
    global result
    tot, citystr, thiscity=cityq.get()
    if thiscity==goal:
        result=citystr+" : : "+str(tot)
        return
    for cty in dict_gn[thiscity]:
        cityq.put((get_fn(citystr+" , "+cty), citystr+" , "+cty, cty))
    expand(cityq)

def main():
    cityq=Q.PriorityQueue()
    thiscity=start
    cityq.put((get_fn(start),start,thiscity))
    expand(cityq)
    print("The A* path with the total is: ")
    print(result)

main()



Prac 2.b RBFS
import queue as Q
from RMP import dict_gn
from RMP import dict_hn

start='Arad'
goal='Bucharest'
result=''

def get_fn(citystr):
    cities=citystr.split(',')
    hn=gn=0
    for ctr in range(0,len(cities)-1):
        gn=gn+dict_gn[cities[ctr]][cities[ctr+1]]
    hn=dict_hn[cities[len(cities)-1]]
    return(hn+gn)

def printout(cityq):
    for i in range(0,cityq.qsize()):
        print(cityq.queue[i])

def expand(cityq):
    global result
    tot,citystr,thiscity=cityq.get()
    nexttot=999
    if not cityq.empty():
        nexttot,nextcitystr,nextthiscity=cityq.queue[0]
    if thiscity==goal and tot<nexttot:
        result=citystr+'::'+str(tot)
        return
    print("Expanded city------------------------------",thiscity)
    print("Second best f(n)------------------------------",nexttot)
    tempq=Q.PriorityQueue()
    for cty in dict_gn[thiscity]:
            tempq.put((get_fn(citystr+','+cty),citystr+','+cty,cty))
    for ctr in range(1,3):
        ctrtot,ctrcitystr,ctrthiscity=tempq.get()
        if ctrtot<nexttot:
            cityq.put((ctrtot,ctrcitystr,ctrthiscity))
        else:
            cityq.put((ctrtot,citystr,thiscity))
            break
    printout(cityq)
    expand(cityq)
def main():
    cityq=Q.PriorityQueue()
    thiscity=start
    cityq.put((999,"NA","NA"))
    cityq.put((get_fn(start),start,thiscity))
    expand(cityq)
    print(result)
main()


Prac 3.b decision tree on test data
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


prac 4 feed foward neural network learning
from doctest import OutputChecker
import numpy as np
class NeuralNetwork():
    def __init__(self):
        np.random.seed()
        self.synaptic_weights=2*np.random.random((3,1))-1
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def  sigmoid_derivative(self,x):
        return x*(1-x)
    def train(self,training_inputs,training_outputs,training_iteration):
        for iteration in range(training_iteration):
            output=self.think(training_inputs)
            error=training_outputs-output
            adjustments=np.dot(training_inputs.T,error*self.sigmoid_derivative(output))
            self.synaptic_weights+=adjustments
    def think(self,inputs):
        inputs=inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))
        return output
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)
    training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    neural_network.train(training_inputs,training_outputs,15000)
    print("Ending Weights After Training : ")
    print(neural_network.synaptic_weights)
    user_input_one=str(input("User Input One: "))
    user_input_two=str(input("User Input Two: "))
    user_input_three=str(input("User Input Three: "))
    print("Considering New Situation: ",  user_input_one, user_input_two, user_input_three)
    print("New Output Data : ")
    print(neural_network.think(np.array([user_input_one,user_input_two,user_input_three])))



pract 5 SVM
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

prac 6 AdaBoost
from warnings import filterwarnings
filterwarnings("ignore")

import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "prac 6/pima-indians-diabetes.data.csv"
names =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = pandas.read_csv(url,names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
model =AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
result = model_selection.cross_val_score(model,X,Y)
print(result.mean())


prac 7 naive base
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


prac 8 KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

plt.style.use('ggplot')

# Load the data
df = pd.read_csv("diabetes.csv")
print(df.head(),"\n")
print(df.shape,"\n")
print(df.dtypes,"\n")

# Prepare the data
x = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# KNN model training and evaluation
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)

# Plotting the accuracy for different values of k
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Predict probabilities for ROC AUC score calculation
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(x_train, y_train)
y_pred_prob = knn.predict_proba(x_test)[:, 1]

# Grid Search for optimal hyperparameters
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
print("\n\n",knn_cv.fit(x, y))

print("\n\n",knn_cv.best_score_)
print("\n\n",knn_cv.best_params_)


prac 9 associatioan mining rule 
from warnings import filterwarnings
filterwarnings("ignore")
    
import numpy as np
import  pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from apyori import apriori

df = pd.read_csv('prac 9\Groceries_dataset.csv')
print(df.head())
print(df.isnull().any())
all_products=df['itemDescription'].unique()
print("Total products: {}".format(len(all_products)))
def ditribution_plot(x,y,name=None,xaxis=None,yaxis=None):
    fig = go.Figure([go.Bar(x=x,y=y)])
    fig.update_layout(title_text=name,xaxis_title=xaxis,yaxis_title=yaxis)
    print(fig.show())

x = df['itemDescription'].value_counts()
x = x.sort_values(ascending = False)
x = x[:10]
ditribution_plot(x=x.index,y=x.values,yaxis="Count",xaxis="Products")

one_hot = pd.get_dummies(df['itemDescription'])
df.drop('itemDescription',inplace=True, axis=1)
df = df.join(one_hot)
print(df.head())
records=df.groupby(["Member_number","Date"])[all_products[:]].apply(sum)
records=records.reset_index()[all_products]

def get_Pnames(x):
    for product in all_products:
        if x[product]>0:
            x[product]=product
    return x
records=records.apply(get_Pnames, axis=1)
print(records.head())
x=records.values
x=[sub[~(sub==0)].tolist() for sub in x if sub[sub != 0].tolist()]
transactions=x

print(transactions[0:10])

rules = apriori(transactions,min_support=0.00030,min_confidance=0.05,min_lift=3,min_length=2,target="rules")
association_results=list(rules)

for item in association_results:
    pair=item[0]
    items=[x for x in pair]
    print("Rule: "+items[0] + " -> " + items[1])
    print("Support: "+str(item[1]))
    print("Confidence: "+str(item[2][0][2]))
    print("Lift: "+str(item[2][0][3]))
    print("========================================================\n\n")


''' 