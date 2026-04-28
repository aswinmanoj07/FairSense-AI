import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

# ---------------- CONFIG ----------------
SENSITIVE_FEATURES = ["Gender", "Race", "Age"]

st.set_page_config(page_title="FairSense AI", layout="wide")

st.markdown("""
# 🚀 FairSense AI  
### 🧠 Enterprise AI Fairness Auditor  
#### Detect • Explain • Decide • Fix Bias in Real-Time
""")

st.info("Fairness is evaluated primarily on sensitive attributes (e.g., Gender, Race, Age)")

# ---------------- CORE ----------------
def detect_bias(df, target):
    result = {}

    # -------- HANDLE TARGET --------
    if df[target].dtype == 'object':
        unique_vals = df[target].dropna().unique()

        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[target] = df[target].map(mapping)
        else:
            st.warning("Target must be binary for fairness analysis")
            return {}
    else:
        st.warning("Numeric target detected → converted to binary using median split")
        threshold = df[target].median()
        df[target] = (df[target] > threshold).astype(int)

    # -------- BIAS --------
    for col in df.columns:
        if col != target and df[col].dtype == 'object':

            # Ignore ID columns
            if df[col].nunique() == len(df):
                continue

            groups = df.groupby(col)[target].mean()

            if len(groups) < 2:
                continue

            max_v, min_v = groups.max(), groups.min()
            spd = abs(max_v - min_v)
            di = min_v / max_v if max_v != 0 else 0

            result[col] = {"SPD": round(spd, 3), "DI": round(di, 3)}

    return result


def fairness_score(bias):
    if not bias:
        return 1.0
    return round(max(0, 1 - sum(v["SPD"] for v in bias.values()) / len(bias)), 3)


# 🔥 FIXED DECISION (SENSITIVE ONLY)
def deployment_decision(bias, score):
    sensitive_bias = {f: v for f, v in bias.items() if f in SENSITIVE_FEATURES}

    if sensitive_bias:
        max_feature = max(sensitive_bias, key=lambda x: sensitive_bias[x]["SPD"])
        max_spd = sensitive_bias[max_feature]["SPD"]

        if max_spd > 0.2:
            return "❌ DO NOT DEPLOY", f"High bias in sensitive feature: {max_feature}"

    if score < 0.8:
        return "⚠️ DEPLOY WITH CAUTION", "Moderate fairness"
    else:
        return "✅ SAFE TO DEPLOY", "Low bias risk"


def confidence_level(score):
    return round(score * 100, 1)


# 🔥 IMPROVED THRESHOLDS
def validate_bias(bias):
    msgs = []
    for f, v in bias.items():
        label = "(Sensitive)" if f in SENSITIVE_FEATURES else "(Non-sensitive)"

        if v["SPD"] >= 0.8:
            msgs.append(f"{f} {label}: Extreme bias")
        elif v["SPD"] > 0.2:
            msgs.append(f"{f} {label}: High bias")
        elif v["SPD"] > 0.15:
            msgs.append(f"{f} {label}: Moderate bias")
        else:
            msgs.append(f"{f} {label}: Acceptable")
    return msgs


def simulate_fix(bias):
    improved = {}
    for k, v in bias.items():
        improved[k] = {
            "SPD": round(v["SPD"] * 0.6, 3),
            "DI": round(min(1, v["DI"] + 0.2), 3)
        }
    return improved


# 🔥 AI REPORT (ALIGNED)
def explain_bias(bias):
    explanation = "🧠 FairSense AI Auditor Report\n\n"

    critical = False

    for f, v in bias.items():
        tag = "(Sensitive)" if f in SENSITIVE_FEATURES else "(Non-sensitive)"

        if v["SPD"] >= 0.8:
            explanation += f"{f} {tag}: Critical bias detected\n"
            if f in SENSITIVE_FEATURES:
                critical = True

        elif v["SPD"] > 0.2:
            explanation += f"{f} {tag}: High bias\n"
            if f in SENSITIVE_FEATURES:
                critical = True

        elif v["SPD"] > 0.15:
            explanation += f"{f} {tag}: Moderate bias\n"
        else:
            explanation += f"{f} {tag}: Low bias\n"

    explanation += "\n💡 Insight:\nFairness is evaluated primarily on sensitive attributes.\n"

    explanation += """
Risks:
- Discriminatory outcomes
- Compliance violations
- Loss of trust

Fixes:
- Rebalance dataset
- Remove sensitive attributes
- Apply fairness-aware models
"""

    if critical:
        explanation += "\n🚫 Recommendation: Do NOT deploy"
    else:
        explanation += "\n✅ Recommendation: Safe to deploy"

    return explanation


# ---------------- UI ----------------
file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    target = st.selectbox("Select Target", df.columns)

    if target.lower() in ["gender", "race"]:
        st.warning("Sensitive attribute cannot be target")
        st.stop()

    if st.button("Run Audit"):
        bias = detect_bias(df.copy(), target)

        if not bias:
            st.error("No valid features or invalid target")
            st.stop()

        score = fairness_score(bias)
        decision, reason = deployment_decision(bias, score)
        confidence = confidence_level(score)

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Fairness Score", score)
        c2.metric("Confidence", f"{confidence}%")
        c3.metric("Bias Features", len(bias))

        # Decision
        st.subheader("🚦 Deployment Decision")
        st.write(decision + " — " + reason)

        # Bias Severity
        st.subheader("Bias Severity Assessment")
        for msg in validate_bias(bias):
            st.write("•", msg)

        # Metrics
        st.subheader("Bias Metrics")
        for f, v in bias.items():
            label = "(Sensitive)" if f in SENSITIVE_FEATURES else "(Non-sensitive)"
            st.write(f"{f} {label}: SPD={v['SPD']} | DI={v['DI']}")

        # Graph
        st.subheader("Visualization")
        plt.figure()
        plt.bar(list(bias.keys()), [v["SPD"] for v in bias.values()])
        plt.ylabel("Bias Score (SPD)")
        plt.title("Bias Severity")
        plt.ylim(0, 1)
        st.pyplot(plt)

        # Simulation
        st.subheader("Before vs After Fix")
        st.write("Original:", bias)
        st.write("Improved:", simulate_fix(bias))

        # AI
        st.subheader("AI Fairness Report")
        st.write(explain_bias(bias))