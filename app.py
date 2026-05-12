import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ----------------------------
# PAGE SETTINGS
# ----------------------------
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ----------------------------
# TITLE
# ----------------------------
st.title("🎫 Support Ticket Classification System")

st.markdown("""
This Machine Learning system predicts customer support ticket categories using NLP and Machine Learning.
""")

# ----------------------------
# LOAD DATASET
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_support_tickets_200k.csv")

    df = df[['issue_description', 'category', 'priority']]

    df = df.dropna()

    return df


df = load_data()

# ----------------------------
# TEXT CLEANING
# ----------------------------
def clean_text(text):
    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    return " ".join(words)

# Apply cleaning
df['cleaned_text'] = df['issue_description'].apply(clean_text)

# ----------------------------
# FEATURES + TARGET
# ----------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['cleaned_text'])

y = df['category']

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

# ----------------------------
# MODEL ACCURACY
# ----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

# ----------------------------
# LAYOUT COLUMNS
# ----------------------------
left_col, right_col = st.columns([2, 1])

# =========================================================
# LEFT SIDE
# =========================================================
with left_col:

    st.success(f"✅ Model Accuracy: {accuracy:.2f}")

    st.subheader("Enter Support Ticket")

    user_input = st.text_area(
        "Type customer issue here:",
        height=150,
        placeholder="Example: Payment failed but money deducted"
    )

    if st.button("Predict Category"):

        if user_input.strip() == "":
            st.warning("Please enter a support issue.")

        else:
            cleaned_input = clean_text(user_input)

            vector_input = vectorizer.transform([cleaned_input])

            prediction = model.predict(vector_input)

            # Priority Logic
            issue = user_input.lower()

            if "payment" in issue or "refund" in issue or "failed" in issue:
                priority = "High 🔴"

            elif "login" in issue or "password" in issue:
                priority = "Medium 🟠"

            else:
                priority = "Low 🟢"

            st.markdown("---")

            st.subheader("Prediction Result")

            st.write(f"### 📂 Category: {prediction[0]}")

            st.write(f"### ⚡ Priority: {priority}")

# =========================================================
# RIGHT SIDE GRAPH
# =========================================================
with right_col:

    st.subheader("📊 Ticket Category Distribution")

    category_counts = df['category'].value_counts().head(5)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(category_counts.index, category_counts.values)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("⚡ Priority Distribution")

    priority_counts = df['priority'].value_counts()

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    ax2.pie(
        priority_counts.values,
        labels=priority_counts.index,
        autopct='%1.1f%%'
    )

    st.pyplot(fig2)

# ----------------------------
# DATASET PREVIEW
# ----------------------------
st.markdown("---")

if st.checkbox("Show Dataset Sample"):
    st.dataframe(df.head())

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")

st.caption("Built using Python, Streamlit, Scikit-learn, and NLP")