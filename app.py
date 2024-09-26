import base64
import pickle

import numpy
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.svm
import sklearn.tree
import streamlit as st

st.title("Credit Card Fraud Detection")


# Loading Models

DecisionTree: sklearn.tree.DecisionTreeClassifier = pickle.load(
    open("models/dt.bin", "rb")
)
GradientBoosting: sklearn.ensemble.GradientBoostingClassifier = pickle.load(
    open("models/gbc.bin", "rb")
)
LogisticRegression: sklearn.linear_model.LogisticRegression = pickle.load(
    open("models/logistic.bin", "rb")
)
RandomForest: sklearn.ensemble.RandomForestClassifier = pickle.load(
    open("models/rf.bin", "rb")
)
LinearSVC: sklearn.svm.LinearSVC = pickle.load(open("models/svc.bin", "rb"))

models = {
    "Decision Tree Classifier": DecisionTree,
    "Gradient Boosting Classifier": GradientBoosting,
    "Logistic Regression": LogisticRegression,
    "Random Forest Classifier": RandomForest,
    "LinearSVC": LinearSVC,
}

model = st.selectbox(
    label="Model",
    placeholder="Select a model",
    options=[
        "Decision Tree Classifier",
        "Gradient Boosting Classifier",
        "Logistic Regression",
        "Random Forest Classifier",
        "LinearSVC",
    ],
)

model = models.get(model)

data = st.selectbox(
    label="select data", options=open("validation_data.csv").readlines()
)
data = list(map(float, data.split(",")))
data = numpy.asarray(data).reshape(1, -1)
res = 0
if st.button("predict"):
    res = (model.predict(data))[0]
    pred = {0: "Non fraudulent transaction", 1: "Fraudulent transaction"}
    if not res:
        st.success(pred.get(res))
    else:
        st.error(pred.get(res))
