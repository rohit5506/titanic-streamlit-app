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

# Load Titanic data
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

# Model Training with hyperparameters and cache
@st.cache_data(show_spinner=False)
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
    
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, y_train)
    return model

# Sidebar Configuration + Tooltips
st.sidebar.header("Model Configuration")
model_name = st.sidebar.selectbox("Choose classification model", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
                                  help="Select the ML algorithm to use for prediction")

hyperparams = {}
if model_name == "Random Forest":
    hyperparams["n_estimators"] = st.sidebar.slider("Number of Trees", 50, 300, 100, step=10)
    hyperparams["max_depth"] = st.sidebar.slider("Max Tree Depth", 3, 30, 10)
elif model_name == "Decision Tree":
    hyperparams["max_depth"] = st.sidebar.slider("Max Tree Depth", 1, 30, 5)

model = train_model(model_name, hyperparams)

# Show model accuracy and confusion matrix
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.title("Titanic Survival Prediction - Enhanced")

st.markdown(f"**Model:** {model_name}")
st.markdown(f"**Train Accuracy:** {train_acc:.3f}")
st.markdown(f"**Test Accuracy:** {test_acc:.3f}")

st.subheader("Confusion Matrix (Test Data)")
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
im = ax.matshow(cm, cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# User Input Section
st.sidebar.header("Passenger Details for Prediction")
input_dict = {
    "Pclass": st.sidebar.selectbox("Passenger Class", [1, 2, 3], help="1=1st, 2=2nd, 3=3rd class"),
    "Sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female"),
    "Age": st.sidebar.slider("Age", 0, 100, 30),
    "SibSp": st.sidebar.slider("Siblings/Spouses Aboard", 0, 8),
    "Parch": st.sidebar.slider("Parents/Children Aboard", 0, 8),
    "Fare": st.sidebar.slider("Fare", 0.0, 600.0, 32.20, step=0.1),
    "Embarked_Q": 1 if st.sidebar.selectbox("Embarked at Queenstown?", ["No", "Yes"]) == "Yes" else 0,
    "Embarked_S": 1 if st.sidebar.selectbox("Embarked at Southampton?", ["No", "Yes"]) == "Yes" else 0,
}

user_df = pd.DataFrame([input_dict])
user_df[features] = scaler.transform(user_df[features])

# Predict button with progress bar and explanation
if st.button("Predict Survival"):
    with st.spinner("Making prediction..."):
        prediction = model.predict(user_df)[0]
        prob = model.predict_proba(user_df)[0][1] if hasattr(model, "predict_proba") else None
    
    st.markdown(f"## Prediction: {'Survived' if prediction==1 else 'Did Not Survive'}")
    if prob is not None:
        st.write(f"Survival Probability: {prob:.2f}")
    
    # Global Feature Importance using Permutation Importance
    st.subheader("Feature Importance (Global via Permutation)")
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm_result.importances_mean
    }).sort_values("Importance", ascending=False)
    st.bar_chart(perm_df.set_index("Feature")["Importance"].round(3))

# Batch Prediction Upload
st.header("Batch Prediction - Upload CSV")
uploaded_file = st.file_uploader("Upload CSV (must contain Titanic features)", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    missing_cols = [f for f in features if f not in batch_df.columns]
    if missing_cols:
        st.error(f"Your CSV is missing these required columns: {missing_cols}")
    else:
        batch_df[features] = scaler.transform(batch_df[features])
        batch_preds = model.predict(batch_df[features])
        batch_df["Predicted_Survived"] = batch_preds
        st.write(batch_df.head())
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name="batch_predictions.csv")

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
st.write("Survival Rate by Passenger Class")
fig1, ax1 = plt.subplots()
df.groupby("Pclass")["Survived"].mean().plot(kind="bar", ax=ax1)
ax1.set_ylabel("Survival Rate")
st.pyplot(fig1)

st.write("Survival Rate by Sex")
fig2, ax2 = plt.subplots()
df.groupby("Sex")["Survived"].mean().plot(kind="bar", color=['blue', 'orange'], ax=ax2)
ax2.set_ylabel("Survival Rate")
ax2.set_xticklabels(["Male", "Female"], rotation=0)
st.pyplot(fig2)

# Footer message
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | [Live Demo](https://titanic-test-app-qwfv.streamlit.app/)")
