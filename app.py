import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_data
def train_model(model_name, hyperparams):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=hyperparams["n_estimators"],
                                       max_depth=hyperparams["max_depth"],
                                       random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=hyperparams["max_depth"])
    elif model_name == "SVM":
        model = SVC(probability=True)
    else:
        model = RandomForestClassifier(random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Sidebar model selection and hyperparameters
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"])

# Hyperparameters UI depending on model
hyperparams = {}
if model_name == "Random Forest":
    hyperparams["n_estimators"] = st.sidebar.slider("Number of Trees", 50, 300, 100, step=10)
    hyperparams["max_depth"] = st.sidebar.slider("Max Depth", 3, 30, 10)
elif model_name == "Decision Tree":
    hyperparams["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 5)
else:
    hyperparams = {}

model = train_model(model_name, hyperparams)

# Model performance
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.title("Titanic Survival Prediction - Advanced App")

st.write(f"Model: **{model_name}**")
st.write(f"Train Accuracy: **{train_acc:.3f}**")
st.write(f"Test Accuracy: **{test_acc:.3f}**")

# Confusion matrix plot
st.write("### Confusion Matrix (Test Data)")
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
im = ax.matshow(cm, cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], va='center', ha='center', color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Sidebar user inputs for prediction
st.sidebar.header("Passenger Information Input")
input_dict = {
    "Pclass": st.sidebar.selectbox("Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3]),
    "Sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female"),
    "Age": st.sidebar.slider("Age", 0, 100, 30),
    "SibSp": st.sidebar.slider("Siblings/Spouses aboard", 0, 8, 0),
    "Parch": st.sidebar.slider("Parents/Children aboard", 0, 8, 0),
    "Fare": st.sidebar.slider("Fare", 0.0, 600.0, 32.20),
    "Embarked_Q": 1 if st.sidebar.selectbox("Embarked at Queenstown?", ["No", "Yes"]) == "Yes" else 0,
    "Embarked_S": 1 if st.sidebar.selectbox("Embarked at Southampton?", ["No", "Yes"]) == "Yes" else 0,
}

user_df = pd.DataFrame([input_dict])
user_df[features] = scaler.transform(user_df[features])

if st.button("Predict Survival"):
    prediction = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else None
    st.markdown(f"## Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
    if proba is not None:
        st.write(f"Survival Probability: {proba:.2f}")
    
    # Permutation Feature Importance on test set
    st.write("### Permutation Feature Importance (Global)")
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm_result.importances_mean
    }).sort_values("Importance", ascending=False)
    st.bar_chart(perm_df.set_index("Feature")["Importance"].round(3))

# Batch Prediction Upload
st.header("Batch Prediction Upload")
uploaded_file = st.file_uploader("Upload CSV with passenger data (features only) for batch prediction", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    required_cols = set(features)
    if not required_cols.issubset(set(batch_df.columns)):
        st.error(f"CSV file must contain these columns: {features}")
    else:
        batch_df[features] = scaler.transform(batch_df[features])
        batch_preds = model.predict(batch_df[features])
        batch_df["Predicted_Survived"] = batch_preds
        st.write(batch_df.head())
        
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download predictions CSV", data=csv, file_name="titanic_batch_predictions.csv")

# Exploratory Data Analysis Section
st.header("Exploratory Data Analysis (EDA)")
st.write("Survival Rate by Passenger Class")
eda_fig, ax = plt.subplots()
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=ax)
ax.set_ylabel("Survival Rate")
st.pyplot(eda_fig)

st.write("Survival Rate by Sex")
eda_fig2, ax2 = plt.subplots()
df.groupby("Sex")["Survived"].mean().plot(kind="bar", color=['blue', 'orange'], ax=ax2)
ax2.set_ylabel("Survival Rate")
ax2.set_xticklabels(["Male", "Female"], rotation=0)
st.pyplot(eda_fig2)
