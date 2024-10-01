# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
import nltk
import joblib
nltk.download('stopwords')


sns.set_style('whitegrid')
sns.set_palette('Set2')

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/Jan/bigdata_sms/spam.csv',encoding="latin-1")

df.sample(5)

#removing unwanted columns and renaming the columns
df = df[["v1","v2"]]
df.rename(columns={"v1":"label","v2":"text"},inplace=True)

df.head()

df.info()

df.describe()

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.label.value_counts()

sns.barplot(x=df.label.value_counts().index,y=df.label.value_counts())
sns.histplot(df["text"].apply(len),bins=100,kde=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

df.groupby("label").apply(lambda x: x["text"].apply(len).mean()).plot(
    kind="bar", ax=ax1, title="Average Length of Text",xlabel="type",ylabel="length")

df.groupby("label").apply(lambda x: x["text"].apply(lambda x: len(
    x.split())).mean()).plot(kind="bar", ax=ax2, title="Average Number of Words",xlabel="type",ylabel="words",colormap="Pastel1")

df.groupby("label").apply(lambda x: x["text"].apply(lambda x: len(x.split(
    ". "))).mean()).plot(kind="bar", ax=ax3, title="Average Number of Sentences",xlabel="type",ylabel="sentences")

#import wordcloud
from wordcloud import WordCloud

spam_words = ' '.join(list(df[df['label'] == 'spam']['text']))
spam_wc = WordCloud(width = 800,height = 512,max_words=100,background_color="white").generate(spam_words)

plt.figure(figsize = (5, 6))
plt.title('Spam Word Cloud')
plt.imshow(spam_wc)
plt.axis('off')


ham_words = ' '.join(list(df[df['label'] == 'ham']['text']))
ham_wc = WordCloud(width = 800,height = 512,max_words=100,background_color="white").generate(ham_words)

plt.figure(figsize = (5, 6))
plt.title('Ham Word Cloud')
plt.imshow(ham_wc)
plt.axis('off')

df["text"].sample(5)

stopwd = stopwords.words('english')
def clean_text(text):

    text= text.lower() # Lowercasing the text
    text = re.sub('-',' ',text.lower())   # Replacing `x-x` as `x x`
    text = re.sub(r'http\S+', '', text) # Removing Links
    text = re.sub(f'[{string.punctuation}]', '', text) # Remove punctuations
    text = re.sub(r'\s+', ' ', text) # Removing unnecessary spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Removing single characters
    
    words = nltk.tokenize.word_tokenize(text,language="english", preserve_line=True)
    text = " ".join([i for i in words if i not in stopwd and len(i)>2]) # Removing the stop words

    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)
df.head()

X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["label"], test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.head()

vectorizer = CountVectorizer(stop_words='english',lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
X_train_vectorized.shape, X_test_vectorized.shape

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
pd.DataFrame(y_train).value_counts().plot(kind="bar")

sampler = RandomOverSampler(random_state=42)
# We will pass to it the output of Vectorizer from train data
x_train_resampled, y_train_resampled = sampler.fit_resample(
    X_train_vectorized, y_train)
pd.DataFrame(y_train_resampled).value_counts().plot(kind="bar")

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000,solver="sag",tol=0.001,max_iter=500,random_state=15)
lr.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", lr.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", lr.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, lr.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, lr.predict(X_test_vectorized)))
from sklearn.naive_bayes import MultinomialNB

cnb = MultinomialNB(alpha=0.1)
cnb.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", cnb.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", cnb.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, cnb.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, cnb.predict(X_test_vectorized)))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", rf.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", rf.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, rf.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, rf.predict(X_test_vectorized)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", dt.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", dt.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, dt.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, dt.predict(X_test_vectorized)))
from sklearn.svm import SVC

svc = SVC(random_state=42)
svc.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", svc.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", svc.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, svc.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, svc.predict(X_test_vectorized)))
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.7,max_depth=7,n_estimators=200)
xgb.fit(x_train_resampled,y_train_resampled)
print("Train Accuracy: ", xgb.score(x_train_resampled, y_train_resampled))
print("Test Accuracy: ", xgb.score(X_test_vectorized, y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, xgb.predict(x_train_resampled)))
print("Test Precision: ", precision_score(y_test, xgb.predict(X_test_vectorized)))

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(x_train_resampled.toarray(),y_train_resampled)
print("Train Accuracy: ", lgbm.score(x_train_resampled.toarray(), y_train_resampled))
print("Test Accuracy: ", lgbm.score(X_test_vectorized.toarray(), y_test))
print("Train Precision: ", precision_score(
    y_train_resampled, lgbm.predict(x_train_resampled.toarray())))
print("Test Precision: ", precision_score(y_test, lgbm.predict(X_test_vectorized.toarray())))

y_pred_train_lr = lr.predict(x_train_resampled)
y_pred_test_lr = lr.predict(X_test_vectorized)

y_pred_train_cnb = cnb.predict(x_train_resampled)
y_pred_test_cnb = cnb.predict(X_test_vectorized)

y_pred_train_rf = rf.predict(x_train_resampled)
y_pred_test_rf = rf.predict(X_test_vectorized)

y_pred_train_dt = dt.predict(x_train_resampled)
y_pred_test_dt = dt.predict(X_test_vectorized)

y_pred_train_svc = svc.predict(x_train_resampled)
y_pred_test_svc = svc.predict(X_test_vectorized)

y_pred_train_xgb = xgb.predict(x_train_resampled)
y_pred_test_xgb = xgb.predict(X_test_vectorized)

y_pred_train_lgbm = lgbm.predict(x_train_resampled.toarray())
y_pred_test_lgbm = lgbm.predict(X_test_vectorized.toarray())

from sklearn.metrics import roc_auc_score,roc_curve

modelsdict = {"LR":lr,"CNB":cnb,"RF":rf,"DT":dt,"SVC":svc,"XGB":xgb,"LGBM":lgbm}

scoresdict = {}

for key,value in modelsdict.items():

    Train_ACC=accuracy_score(y_train_resampled,value.predict(x_train_resampled.toarray()))
    Train_Prec=precision_score(y_train_resampled,value.predict(x_train_resampled.toarray()))
    Test_ACC=accuracy_score(y_test,value.predict(X_test_vectorized.toarray()))
    Test_Prec=precision_score(y_test,value.predict(X_test_vectorized.toarray()))

    scoresdict[key] = [Train_ACC,Train_Prec,Test_ACC,Test_Prec]

scoresdf = pd.DataFrame(scoresdict,index=["Train_ACC","Train_Prec","Test_ACC","Test_Prec"]).T

scoresdf.sort_values(by="Test_ACC",ascending=False)
scoresdf.sort_values(by="Test_Prec",ascending=False)
scoresdf.plot(kind="bar",title="Accuracy and Precision Scores",xlabel="Models",ylabel="Accuracy")

plt.figure(figsize=(8,5))

for key,value in modelsdict.items():
    try:
        fpr,tpr,thresholds = roc_curve(y_test,value.predict_proba(X_test_vectorized.toarray())[:,1])
    except:
        fpr,tpr,thresholds = roc_curve(y_test,value.predict(X_test_vectorized.toarray()))
    plt.plot(fpr,tpr,label=key)
plt.legend()

cnf_lr = confusion_matrix(y_test,y_pred_test_lr)
cnf_cnb = confusion_matrix(y_test,y_pred_test_cnb)
cnf_rf = confusion_matrix(y_test,y_pred_test_rf)
cnf_dt = confusion_matrix(y_test,y_pred_test_dt)
cnf_svc = confusion_matrix(y_test,y_pred_test_svc)
cnf_xgb = confusion_matrix(y_test,y_pred_test_xgb)
cnf_lgbm = confusion_matrix(y_test,y_pred_test_lgbm)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.heatmap(cnf_lr,annot=True,fmt="d")
plt.title("Logistic Regression")
plt.subplot(1,3,2)
sns.heatmap(cnf_cnb,annot=True,fmt="d")
plt.title("Complement Naive Bayes")
plt.subplot(1,3,3)
sns.heatmap(cnf_rf,annot=True,fmt="d")
plt.title("Random Forest")

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.heatmap(cnf_dt,annot=True,fmt="d")
plt.title("Decision Tree")
plt.subplot(1,3,2)
sns.heatmap(cnf_svc,annot=True,fmt="d")
plt.title("Support Vector Classifier")
plt.subplot(1,3,3)
sns.heatmap(cnf_xgb,annot=True,fmt="d")
plt.title("XGBoost")

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.heatmap(cnf_lgbm,annot=True,fmt="d")
plt.title("LightGBM")

joblib.dump(rf, 'rf.joblib')
joblib.dump(le, 'le.joblib')
joblib.dump(lgbm, 'lgbm.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
