import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv("/kaggle/input/wine-quality-dataset/WineQT.csv")
data.head()
data.info()
data.describe()
data.hist(bins=20, figsize=(10, 10))
plt.show()
data.columns
# Z-Score Method: 
from scipy.stats import zscore
# Calculate Z-scores
z_scores = zscore(data["sulphates"])
z_scores.mean()
threshold = (100 -data["sulphates"].mean() )/data["sulphates"].std()
outliers_mask = abs(z_scores) > threshold

# Remove rows with outliers
data = data[~outliers_mask]
data["quality"].value_counts()
data["quality"]=data["quality"].map({3:0,4:1,5:2,6:3,7:4,8:5})
data.drop("Id",axis=1,inplace=True)
data.isnull().sum()
data.columns
sns.heatmap(data.corr())
data.drop("free sulfur dioxide",axis=1,inplace=True)
data.duplicated().sum()
data.quality.value_counts()
data["quality"]=[1 if x< 3 else 0 for x in data.quality]
data.quality.value_counts()
from sklearn.model_selection import train_test_split
x=data.drop("quality",axis=1)
y=data["quality"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y)
ytrain.shape
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report
mse = confusion_matrix(ytest, ypred)
mse
print(classification_report(ytest, ypred))
