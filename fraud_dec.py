import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
df=pd.read_csv("AIML Dataset.csv")
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
df.head()
df.shape
df.describe()
df["isFraud"].value_counts()
sns.countplot(x=df["isFraud"].head())
plt.show()
df["isFlaggedFraud"].value_counts()
df.columns
df["type"].value_counts().plot(kind="bar",title="Tranction types",color="skyblue")
plt.xlabel("Tranction type")
plt.ylabel("Count")
plt.show()
df[['oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest']]
fraud_by_type=df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
fraud_by_type.plot(kind="bar")
df["amount"].describe().astype(int)
sns.histplot(np.log1p(df["amount"]),bins=100,kde=True,color="green")
plt.title("Tranction amount histplot (log scale)")
plt.xlabel("log(Amount+1)")
plt.show()
sns.boxplot(data=df[df["amount"]<50000],x="isFraud",y="amount")
plt.title("Amount vs Fraud (under 50000)")
plt.show()
df.columns
df["balancediffOrig"]=df["oldbalanceOrg"]-df["newbalanceOrig"]
df["balancediffDest"]=df["newbalanceDest"]-df["oldbalanceDest"]
(df["balancediffOrig"]<0).sum()
(df["balancediffDest"]<0).sum()
df.head()
df.drop("step",axis=1,inplace=True)
df.head()
top_senders=df["nameOrig"].value_counts().head(10)
top_senders
top_receivers=df["nameDest"].value_counts().head(10)
top_receivers
fraud_users=df[df["isFraud"]==1]["nameOrig"].value_counts().head(10)
fraud_users
fraud_types=df[df["type"].isin(["TRANSFER","CASH_OUT"])]

fraud_types["type"].value_counts()
sns.countplot(data=fraud_types,x="type",hue="isFraud")
plt.title("Fraud Distribution in tarnsfer and cash_out")
plt.show()
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation matrix")
plt.show()
zero_after_transfer=df[(df["oldbalanceOrg"]>0)&(df["newbalanceOrig"]==0)&(df["type"].isin(["TRANSFER","CASH_OUT"]))]
len(zero_after_transfer)
zero_after_transfer.head()
df["isFraud"].value_counts()
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
df.head()
df.columns
df_model=df.drop(["nameOrig","nameDest","isFlaggedFraud"],axis=1)
df_model
catergory=["type"]
numeric=["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
X=df_model.drop("isFraud",axis=1)
y=df_model["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numeric),
        ("cat",OneHotEncoder(drop="first"),catergory)
    ],remainder="drop"
)
pipeline=Pipeline(
    [
        ("prep",preprocessor),
        ("clf",LogisticRegression(class_weight="balanced",max_iter=1000))
    ]
)
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)

print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
pipeline.score(X_test,y_test)

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train,y_train)
import joblib
joblib.dump(pipeline,"frd_dect.pkl")
