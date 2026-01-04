import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Wrestling Match Predictor",
    page_icon="🤼",
    layout="centered"
)

st.title("Wrestling Match Winner Predictor")
st.write(
    "Select two wrestlers and the model will predict "
    "who is more likely to win based on historical performance."
)

# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
@st.cache_data
def load_features():
    return pd.read_csv("data/wrestler_features_2023.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/match_winner_model.pkl")

features = load_features()
model = load_model()

# Feature order EXACTLY as model expects
FEATURE_COLUMNS = list(model.feature_names_in_)

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def predict_match(w1, w2):
    f1 = features[features["wrestlers"] == w1].iloc[0]
    f2 = features[features["wrestlers"] == w2].iloc[0]

    input_data = pd.DataFrame([{
        "score_diff": f1["score"] - f2["score"],
        "win_rate_diff": f1["win_rate"] - f2["win_rate"],
        "title_wins_diff": f1["title_wins"] - f2["title_wins"],
        "matches_diff": f1["total_matches"] - f2["total_matches"]
    }])

    # Force correct feature order
    input_data = input_data[FEATURE_COLUMNS]

    prob = model.predict_proba(input_data)[0][1]

    if prob >= 0.5:
        return w1, prob
    else:
        return w2, 1 - prob

# --------------------------------------------------
# UI - Wrestler Selection
# --------------------------------------------------
wrestler_list = sorted(features["wrestlers"].unique())

col1, col2 = st.columns(2)

with col1:
    wrestler_a = st.selectbox("Select Wrestler A", wrestler_list)

with col2:
    wrestler_b = st.selectbox("Select Wrestler B", wrestler_list)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
st.markdown("---")

if wrestler_a == wrestler_b:
    st.warning("Please select two different wrestlers.")
else:
    if st.button("Predict Winner"):
        winner, prob = predict_match(wrestler_a, wrestler_b)

        st.success(f"**Predicted Winner: {winner}**")
        st.metric("Winning Probability", f"{prob:.2%}")

        st.progress(min(prob, 1.0))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Project by Anik => Built using Machine Learning • Streamlit • Python")
