import csv
import pandas
import statistics
import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from collections import defaultdict
import math
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import scikitplot as skplt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

num = []
df1 = pd.read_csv('C:\\Users\\Seemorgh\\Desktop\\dadekavi\\diabetes.csv')

### 1-Preproccesing *************************************

#### Replace the zero values of the 'Glucose','BloodPressure','SkinThickness','Insulin','BMI' attributes with one 

for l in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df1[l] = df1[l].replace(0,df1[l].mean())

#### We determined that the number of pregnancies has little connection with DM. Therefore, we transformed this numeric attribute into a nominal attribute 
###  The value 0 indicates non-pregnant and 1 indicates pregnant 

for k in df1['Pregnancies']:  
    if k != 0:
        df1['Pregnancies'] = df1['Pregnancies'].replace(k,1)

        
### 2-Normalization ************************************
        
#Normalize columns = Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age with Z-Score
        
for col in df1.columns[1:-1]:
    df1[col] = (df1[col] - df1[col].mean())/df1[col].std(ddof=0)
    #print(col,'\n',df1[col])  

df1.to_csv(r'C:\Users\Seemorgh\Desktop\dadekavi\myfile.csv', index = False)

### 3-K-Means Algorithm *****************************************

# composing data set
Xd = df1.drop('Outcome', 1)
yd = df1['Outcome']
xxd=np.array(Xd)
km = KMeans(n_clusters = 2)
clusters=km.fit_predict(xxd)

# obtaining the centers of the clusters
centroids = km.cluster_centers_

# points array will be used to reach the index easy
points = np.empty((0,len(xxd[0])), float)

# distances will be used to calculate outliers
distances = np.empty((0,len(xxd[0])), float)

# getting points and distances
for i, center_elem in enumerate(centroids):
    # cdist is used to calculate the distance between center and other points
    distances = np.append(distances, cdist([center_elem],xxd[clusters == i], 'euclidean')) 
    points = np.append(points, xxd[clusters == i], axis=0)
percentile = 77

# getting outliers whose distances are greater than some percentile
outliers = points[np.where(distances > np.percentile(distances, percentile))]
#print("outliers",outliers)
#print("len:",len(outliers))
with open("outlier_data.csv", 'w') as f: 
    f.write("\n".join(["{},{},{},{},{},{},{},{}".format(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) for p in outliers]))
    
### Remove Outliers from df1
for indexx in outliers:
    num.append((df1[(df1.Pregnancies == indexx[0]) & (df1.Glucose == indexx[1]) & (df1.BloodPressure == indexx[2]) & (df1.SkinThickness == indexx[3]) &
                    (df1.Insulin == indexx[4]) & (df1.BMI == indexx[5]) & (df1.DiabetesPedigreeFunction == indexx[6]) & (df1.Age == indexx[7])].index)[0])
df1 = df1.drop(num, 0)

### 4-Regeression Logestic *********************************
features = df1.drop(["Outcome"], axis=1)
lbl = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(features, lbl, test_size=0.2, random_state=29)
#print("fariba",len(X_train), len(X_test), len(y_train), len(y_test))
logreg = LogisticRegression().fit(X_train, y_train)
#print('X_train, y_train',X_train, y_train)
y_pred = logreg.predict(X_test)
print('model accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))


### 5-Validation *****************************************
confusion_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:",'\n',confusion_matrix)
scores = cross_val_score(logreg,X_train,y_train,scoring='accuracy',cv=10,n_jobs=-1)
print('10-fold cross validation: {:.3f}'.format((mean(scores)*100)))
print('kappa: {:.3f}'.format((cohen_kappa_score(y_test, y_pred))))
print('mcc: {:.3f}'.format((matthews_corrcoef(y_test, y_pred))))
print('precision: {:.3f}'.format((precision_score(y_test, y_pred))))
print('recall: {:.3f}'.format((recall_score(y_test, y_pred))))
fpr, tpr, thresholds =  roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('roc_auc: {:.3f}'.format((roc_auc_score(y_test, y_pred))))
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
print(plt.show())
