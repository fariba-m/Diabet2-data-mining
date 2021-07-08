import csv
import pandas
import statistics
import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from collections import defaultdict
import math
from numpy import mean
from numpy import std
num = []
df1 = pd.read_csv('C:\\Users\\Seemorgh\\Desktop\\dadekavi\\diabetes.csv')
for i in df1:
    print(i,":max is =",df1[i].max(),",","min is =",df1[i].min(),",","Mean is =",df1[i].mean(),",","Median is =",df1[i].median(),",","Mode and its repeated is =",df1[i].value_counts()[df1[i].value_counts() == df1[i].value_counts().max()])
    print("***********************")
#sns.boxplot(y=df1["Pregnancies"],width= 0.2)
#print(plt.show())
#sns.boxplot(y=df1["Glucose"],width= 0.2)
#print(plt.show())
df1.boxplot(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
print(plt.show())
