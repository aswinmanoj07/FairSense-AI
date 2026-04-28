import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv(dotenv_path=".env")

# ---------------- CONFIG ----------------
SENSITIVE_FEATURES = ["Gender", "Race", "Age"]

st.set_page_config(
    page_title="FairSense AI",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 FairSense AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "About"
    ]
)

# ---------------- ABOUT PAGE ----------------
if page == "About":

    st.title("📘 About FairSense AI")

    st.write("""
    FairSense AI is an enterprise-level AI fairness auditing system
    designed to detect, explain, and simulate mitigation of bias
    in machine learning datasets.

    The platform evaluates fairness primarily across sensitive
    attributes such as:
    - Gender
    - Race
    - Age

    Key Features:
    - Bias Detection
    - Fairness Scoring
    - Deployment Decision Support
    - Bias Visualization
    - AI-generated Fairness Reports
    - Bias Mitigation Simulation
    """)

# ---------------- DASHBOARD PAGE ----------------
if page == "Dashboard":

    # ---------------- HEADER ----------------
    st.markdown("""
    # 🚀 FairSense AI
    ### 🧠 Enterprise AI Fairness Auditor
    #### Detect • Explain • Decide • Fix Bias in Real-Time
    """)

    st.info(
        "Fairness is evaluated primarily on sensitive attributes "
        "(e.g., Gender, Race, Age)"
    )

    # ---------------- FUNCTIONS ----------------
    def detect_bias(df, target):

        result = {}

        # -------- HANDLE TARGET --------
        if df[target].dtype == object:

            unique_vals = df[target].dropna().unique()

            if len(unique_vals) == 2:

                mapping = {
                    unique_vals[0]: 0,
                    unique_vals[1]: 1
                }

                df[target] = df[target].map(mapping)

            else:

                st.warning(
                    "Target must be binary for fairness analysis"
                )

                return {}

        else:

            st.warning(
                "Numeric target detected → converted to binary "
                "using median split"
            )

            threshold = df[target].median()

            df[target] = (
                df[target] > threshold
            ).astype(int)

        # -------- BIAS DETECTION --------
        for col in df.columns:

            if (
                col != target
                and df[col].dtype == object
            ):

                # Ignore ID-like columns
                if df[col].nunique() == len(df):
                    continue

                groups = df.groupby(col)[target].mean()

                if len(groups) < 2:
                    continue

                max_v = groups.max()
                min_v = groups.min()

                spd = abs(max_v - min_v)

                di = (
                    min_v / max_v
                    if max_v != 0
                    else 0
                )

                result[col] = {
                    "SPD": round(spd, 3),
                    "DI": round(di, 3)
                }

        return result


    def fairness_score(bias):

        if not bias:
            return 1.0

        avg_bias = (
            sum(v["SPD"] for v in bias.values())
            / len(bias)
        )

        return round(
            max(0, 1 - avg_bias),
            3
        )


    def deployment_decision(bias, score):

        sensitive_bias = {
            f: v
            for f, v in bias.items()
            if f in SENSITIVE_FEATURES
        }

        if sensitive_bias:

            max_feature = max(
                sensitive_bias,
                key=lambda x: sensitive_bias[x]["SPD"]
            )

            max_spd = sensitive_bias[max_feature]["SPD"]

            if max_spd > 0.2:

                return (
                    "❌ DO NOT DEPLOY",
                    f"High bias in sensitive feature: {max_feature}"
                )

        if score < 0.8:

            return (
                "⚠️ DEPLOY WITH CAUTION",
                "Moderate fairness"
            )

        return (
            "✅ SAFE TO DEPLOY",
            "Low bias risk"
        )


    def confidence_level(score):
        return round(score * 100, 1)


    def validate_bias(bias):

        msgs = []

        for f, v in bias.items():

            label = (
                "(Sensitive)"
                if f in SENSITIVE_FEATURES
                else "(Non-sensitive)"
            )

            if v["SPD"] >= 0.8:

                msgs.append(
                    f"{f} {label}: Extreme bias"
                )

            elif v["SPD"] > 0.2:

                msgs.append(
                    f"{f} {label}: High bias"
                )

            elif v["SPD"] > 0.15:

                msgs.append(
                    f"{f} {label}: Moderate bias"
                )

            else:

                msgs.append(
                    f"{f} {label}: Acceptable"
                )

        return msgs


    def simulate_fix(bias):

        improved = {}

        for k, v in bias.items():

            improved[k] = {
                "SPD": round(v["SPD"] * 0.6, 3),
                "DI": round(
                    min(1, v["DI"] + 0.2),
                    3
                )
            }

        return improved


    def explain_bias(bias):

        explanation = (
            "🧠 FairSense AI Auditor Report\n\n"
        )

        critical = False

        for f, v in bias.items():

            tag = (
                "(Sensitive)"
                if f in SENSITIVE_FEATURES
                else "(Non-sensitive)"
            )

            if v["SPD"] >= 0.8:

                explanation += (
                    f"{f} {tag}: Critical bias detected\n"
                )

                if f in SENSITIVE_FEATURES:
                    critical = True

            elif v["SPD"] > 0.2:

                explanation += (
                    f"{f} {tag}: High bias\n"
                )

                if f in SENSITIVE_FEATURES:
                    critical = True

            elif v["SPD"] > 0.15:

                explanation += (
                    f"{f} {tag}: Moderate bias\n"
                )

            else:

                explanation += (
                    f"{f} {tag}: Low bias\n"
                )

        explanation += """
💡 Insight:
Fairness is evaluated primarily on sensitive attributes.

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
            explanation += (
                "\n🚫 Recommendation: Do NOT deploy"
            )

        else:
            explanation += (
                "\n✅ Recommendation: Safe to deploy"
            )

        return explanation

    # ---------------- FILE UPLOAD ----------------
    file = st.file_uploader(
        "Upload Dataset",
        type=["csv"]
    )

    # ---------------- IF FILE EXISTS ----------------
    if file:

        # ---------------- CACHE DATA ----------------
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)

        df = load_data(file)

        # ---------------- DATASET PREVIEW ----------------
        st.subheader("📂 Dataset Preview")

        st.write(
            "Dataset Shape:",
            df.shape
        )

        st.dataframe(df.head())

        # ---------------- TARGET COLUMN ----------------
        target = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # Prevent sensitive target
        if target.lower() in [
            "gender",
            "race",
            "age"
        ]:

            st.warning(
                "Sensitive attribute cannot be target"
            )

            st.stop()

        # ---------------- RUN AUDIT ----------------
        if st.button("Run Audit"):

            bias = detect_bias(
                df.copy(),
                target
            )

            if not bias:

                st.error(
                    "No valid features or invalid target"
                )

                st.stop()

            score = fairness_score(bias)

            decision, reason = deployment_decision(
                bias,
                score
            )

            confidence = confidence_level(score)

            # ---------------- KPIs ----------------
            st.subheader("📈 Key Metrics")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Fairness Score",
                score
            )

            c2.metric(
                "Confidence",
                f"{confidence}%"
            )

            c3.metric(
                "Bias Features",
                len(bias)
            )

            # ---------------- DEPLOYMENT DECISION ----------------
            st.subheader("🚦 Deployment Decision")

            if "DO NOT DEPLOY" in decision:

                st.error(
                    f"{decision} — {reason}"
                )

            elif "CAUTION" in decision:

                st.warning(
                    f"{decision} — {reason}"
                )

            else:

                st.success(
                    f"{decision} — {reason}"
                )

            # ---------------- BIAS VALIDATION ----------------
            st.subheader(
                "⚖️ Bias Severity Assessment"
            )

            for msg in validate_bias(bias):

                st.write("•", msg)

            # ---------------- BIAS METRICS ----------------
            st.subheader("📊 Bias Metrics")

            for f, v in bias.items():

                label = (
                    "(Sensitive)"
                    if f in SENSITIVE_FEATURES
                    else "(Non-sensitive)"
                )

                st.write(
                    f"{f} {label}: "
                    f"SPD={v['SPD']} | "
                    f"DI={v['DI']}"
                )

            # ---------------- GRAPH ----------------
            st.subheader("📉 Bias Visualization")

            colors = [
                "red"
                if v["SPD"] > 0.2
                else "green"
                for v in bias.values()
            ]

            features = list(bias.keys())

            scores = [
                v["SPD"]
                for v in bias.values()
            ]

            fig, ax = plt.subplots(
                figsize=(14, 7)
            )

            ax.bar(
                features,
                scores,
                color=colors
            )

            ax.set_ylabel(
                "Bias Score (SPD)",
                fontsize=12
            )

            ax.set_xlabel(
                "Features",
                fontsize=12
            )

            ax.set_title(
                "Bias Severity Analysis",
                fontsize=14
            )

            ax.set_ylim(0, 1)

            plt.xticks(
                rotation=60,
                ha="right",
                fontsize=10
            )

            plt.tight_layout(pad=3)

            st.pyplot(fig)

            # ---------------- SIMULATION ----------------
            st.subheader(
                "🔧 Before vs After Bias Fix"
            )

            improved_bias = simulate_fix(bias)

            st.write(
                "Original:",
                bias
            )

            st.write(
                "Improved:",
                improved_bias
            )

            # ---------------- AI REPORT ----------------
            st.subheader(
                "🤖 AI Fairness Report"
            )

            report = explain_bias(bias)

            st.write(report)

            # ---------------- DOWNLOAD BUTTON ----------------
            st.download_button(
                label="📥 Download AI Report",
                data=report,
                file_name="fairness_report.txt",
                mime="text/plain"
            )

            # ---------------- FOOTER ----------------
            st.markdown("---")

            st.caption(
                "FairSense AI • Enterprise AI Fairness Auditor"
            )
