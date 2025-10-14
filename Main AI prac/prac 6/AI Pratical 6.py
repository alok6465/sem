from warnings import filterwarnings
filterwarnings("ignore")

import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "pima-indians-diabetes.data.csv"
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
