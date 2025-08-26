import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features, scaler

df, features, scaler = load_data()
X = df[features]
y = df['Survived']

@st.cache_data
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

st.title("Titanic Survival Prediction")

# Sidebar for user input
st.sidebar.header('Passenger Info')
input_dict = {
    "Pclass": st.sidebar.selectbox("Class (1,2,3)", [1,2,3]),
    "Sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==0 else "Female"),
    "Age": st.sidebar.slider("Age", 0, 100, 30),
    "SibSp": st.sidebar.slider("Siblings/Spouses", 0, 5, 0),
    "Parch": st.sidebar.slider("Parents/Children", 0, 5, 0),
    "Fare": st.sidebar.slider("Fare", 0.0, 500.0, 32.2),
    "Embarked_Q": 1 if st.sidebar.selectbox("Embarked Q?", ["No", "Yes"]) == "Yes" else 0,
    "Embarked_S": 1 if st.sidebar.selectbox("Embarked S?", ["No", "Yes"]) == "Yes" else 0,
}

user_df = pd.DataFrame([input_dict])
user_df[features] = scaler.transform(user_df[features])

if st.button("Predict Survival"):
    prediction = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][1]
    st.markdown(f"## Prediction: {'Survived' if prediction==1 else 'Did Not Survive'}")
    st.write(f"Survival Probability: {prob:.2f}")
    
    # Feature Importance Display
    importances = model.feature_importances_
    st.write("### Feature Contributions (importance):")
    st.bar_chart(pd.Series(importances, index=features))

