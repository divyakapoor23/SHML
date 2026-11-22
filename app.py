"""Interactive dashboard for the saved SHML experiments."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FINAL_PATH = RESULTS / "final_test" / "final_test_results.csv"
ROBUST_PATH = RESULTS / "bill_amt1" / "integrated_20_seed_results.csv"
AUC_PATH = RESULTS / "bill_amt1" / "repeated_seed_auc_comparison.csv"


def load_results(path, required_columns):
    try:
        results = pd.read_csv(path)
    except FileNotFoundError:
        return None, f"Missing `{path.relative_to(ROOT)}`."
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None, f"Could not read `{path.relative_to(ROOT)}` as a results CSV."

    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        return None, f"`{path.relative_to(ROOT)}` is missing columns: {', '.join(missing_columns)}."
    if results.empty:
        return None, f"`{path.relative_to(ROOT)}` contains no results."
    return results, None


st.set_page_config(page_title="SHML Experimental Dashboard", layout="wide")
st.title("SHML Experimental Dashboard")

final, final_error = load_results(
    FINAL_PATH,
    {"missingness", "strategy", "f1", "recall", "roc_auc",
     "posthoc_safety_decision", "posthoc_safety_reason"},
)
robust, robust_error = load_results(
    ROBUST_PATH, {"seed", "status", "selected_action", "f1"}
)
auc_seeds, auc_error = load_results(AUC_PATH, {"seed", "median_auc", "zero_auc"})

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Held-out Test", "Safety Policy", "Repeated Seeds"]
)

with tab1:
    st.subheader("What this project actually tests")
    st.markdown(
        "SHML investigates how an **offline machine learning pipeline** handles "
        "**simulated data-quality failures**. Missing `BILL_AMT1` values were "
        "artificially introduced after baseline training; the original dataset "
        "and trained classifier were preserved."
    )
    st.graphviz_chart(r"""
    digraph SHML {
      rankdir=LR;
      node [shape=box style="rounded,filled" fillcolor="#e8f3ec"
            color="#63927b" fontname="Arial" fontsize=11];
      a [label="Healthy data\\n& trained model"];
      b [label="Synthetic NaN\\ninjection"];
      c [label="Missingness\\ndetection"];
      d [label="Rule-based\\ndiagnosis"];
      e [label="Candidate\\nimputation"];
      f [label="F1 / Recall /\\nROC-AUC safety"];
      g [label="Accept / Fallback\\n/ Abstain"];
      a -> b -> c -> d -> e -> f -> g;
    }""", width="stretch")
    x1,x2,x3 = st.columns(3)
    x1.metric("Source missingness", "0%", help="Original predictor values")
    x2.metric("Main severity levels", "0–30%", help="Artificial missingness")
    x3.metric("Robustness masks", "20", help="At 15% missingness, seeds 0–19")
    st.warning("Detected missingness ≠ measured predictive degradation. "
               "Imputation ≠ demonstrated restoration of lost model performance.")

with tab2:
    st.subheader("Held-out test: candidate performance")
    if final is None:
        st.error(final_error)
    else:
        df = final.sort_values("missingness").copy()
        df["Missingness (%)"] = (df["missingness"] * 100).round().astype(int)
        severities = sorted(df["Missingness (%)"].unique())
        metric = st.radio("Evaluation metric", ["F1-score", "ROC-AUC"],
                          horizontal=True, key="test_metric")
        column = "f1" if metric == "F1-score" else "roc_auc"
        f = px.line(df, x="Missingness (%)", y=column, color="strategy",
                    markers=True, title=f"Post-imputation {metric} on held-out test",
                    labels={column: metric, "strategy": "Correction strategy"})
        f.update_xaxes(tickvals=severities)
        if column == "roc_auc":
            healthy = df.loc[df["missingness"].eq(0), "roc_auc"].dropna()
            if not healthy.empty:
                f.add_hline(y=healthy.iloc[0] * (1 - 0.005), line_dash="dot",
                            annotation_text="0.5% relative-loss floor")
        st.plotly_chart(f, width="stretch")
        st.caption("One fixed synthetic missingness mask per severity on the held-out "
                   "test partition. The scores reflect performance after imputation.")
        severity = st.select_slider("Inspect severity", options=severities,
                                    value=severities[-1], format_func=lambda v: f"{v}%")
        filtered = df[df["Missingness (%)"].eq(severity)]
        st.dataframe(filtered[["strategy", "f1", "recall", "roc_auc",
                               "posthoc_safety_decision", "posthoc_safety_reason"]]
                     .round(6), hide_index=True, width="stretch")
        if (len(filtered) == 2
                and filtered["posthoc_safety_decision"].eq("REJECT").all()
                and filtered["posthoc_safety_reason"].eq("ROC-AUC loss exceeds tolerance").all()):
            st.warning(f"At {severity}% artificial missingness, the retrospective audit "
                       "rejects both strategies because ROC-AUC falls below the frozen tolerance.")

with tab3:
    st.subheader("Offline safety-policy logic")
    st.markdown(
        "The integrated policy checks data integrity and allows at most "
        "**10% relative F1 loss**, **10% relative recall loss**, and "
        "**0.5% relative ROC-AUC loss**, relative to the validation baseline. "
        "Zero imputation is tested first; median imputation is a fallback."
    )
    st.markdown("**Possible decisions:** `NO_REPAIR_NEEDED`, "
                "`PREFERRED_ACCEPTED`, `FALLBACK_ACCEPTED`, "
                "`NO_SAFE_RECOVERY_ABSTAIN`.")
    st.info("The label-dependent decisions shown here are offline validation "
            "results. The held-out test safety labels are retrospective audits, "
            "not test-driven strategy selection.")
    if robust is not None:
        order = ["PREFERRED_ACCEPTED", "FALLBACK_ACCEPTED",
                 "NO_SAFE_RECOVERY_ABSTAIN"]
        counts = robust["status"].value_counts().reindex(order, fill_value=0)
        fig = px.bar(x=counts.index.tolist(), y=counts.tolist(),
                     labels={"x":"Safety-policy result", "y":"Seed count"},
                     title="Integrated policy outcomes: 20 corruption masks",
                     text=counts.tolist())
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning(robust_error)

with tab4:
    st.subheader("Repeated-seed evaluation")
    st.markdown(
        "Twenty pseudorandom masks (seeds 0–19) each replace **675 of 4,500** "
        "validation `BILL_AMT1` values. These are repeated perturbations of one "
        "fixed validation set—not twenty independent samples."
    )
    if robust is None:
        st.error(robust_error)
    else:
        select = st.selectbox("Inspect seed", robust["seed"].astype(int).tolist())
        item = robust[robust["seed"] == select]
        if len(item):
            obs = item.iloc[0]
            c1,c2,c3 = st.columns(3)
            c1.metric("Selected action", str(obs["selected_action"]))
            c2.metric("Status", str(obs["status"]))
            c3.metric("F1", f'{obs["f1"]:.4f}' if pd.notna(obs["f1"]) else "—")
            st.dataframe(item, hide_index=True, width="stretch")
        if auc_seeds is not None:
            stacked = auc_seeds.sort_values("seed").melt(id_vars="seed", value_vars=["median_auc","zero_auc"],
                                     var_name="strategy", value_name="ROC-AUC")
            fig = px.line(stacked, x="seed", y="ROC-AUC", color="strategy",
                          markers=True, title="ROC-AUC by corruption mask")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning(auc_error)
        st.caption("20/20 integrated policy decisions matched the recorded notebook "
                   "decisions under the tested configuration.")
st.divider()
st.caption("Scope: controlled synthetic MCAR-like missingness in one feature. "
           "No production monitoring, automatic deployment, persistent rollback, "
           "or established restoration from an untreated degraded baseline.")
