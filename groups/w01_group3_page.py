"""Group 3 - Which customers will come back?"""

import streamlit as st

from ml_base import load_customers
from w01_group3_returning import ReturnPredictor

TEAM_FIRST_NAMES = ('Maciej', 'Maksym', 'Itan', 'Szymon')

st.title("🔁 Which customers will come back?")
team = ", ".join(TEAM_FIRST_NAMES) or "names pending"
st.caption(f"Group 3 · Card 3 · Team: {team}")

customers = load_customers()
predictor = ReturnPredictor(customers).fit()
results = predictor.results_
matrix = predictor.confusion()
gain = 100 * (results["accuracy"] - results["baseline_accuracy"])

st.header("What we built")
st.markdown(
    f"We used **logistic regression** to estimate whether each of **{len(customers):,} customers** "
    "would place another order between **1 September and 9 December 2011**. The model uses "
    "**orders placed** and **days since the last order**, both measured as of 1 September."
)
st.caption(
    f"Trained on {results['n_train']:,} customers and evaluated on {results['n_test']:,} other customers."
)

st.header("Our answer to the board")
accuracy, baseline, improvement = st.columns(3)
accuracy.metric("Model accuracy", f"{results['accuracy']:.2%}")
baseline.metric("Training-majority baseline", f"{results['baseline_accuracy']:.2%}")
improvement.metric("Accuracy gain", f"{gain:+.2f} pp")
st.caption("The baseline predicts that every customer will return, the most common outcome in training.")

st.subheader("What the model got right and wrong")
st.dataframe(
    matrix.rename(
        index={"actual 0": "Did not return", "actual 1": "Returned"},
        columns={"predicted 0": "Predicted no return", "predicted 1": "Predicted return"},
    ),
)
true_nonreturners = int(matrix.loc["actual 0", "predicted 0"])
missed_nonreturners = int(matrix.loc["actual 0", "predicted 1"])
would_return = int(matrix.loc["actual 1", "predicted 0"])
st.markdown(
    f"Sending vouchers to everyone predicted to stay away would reach **{true_nonreturners}** "
    f"people who did stay away and **{would_return}** who came back anyway. That could spend "
    f"discounts on purchases the shop would get at full price. It would also miss "
    f"**{missed_nonreturners}** people who stayed away and might have responded to an offer."
)
st.markdown(
    "**Board recommendation.** Use the score to shortlist customers for a voucher trial. "
    "Randomly give some shortlisted customers a voucher and compare repeat purchases and "
    "profit with the shortlisted customers who received none."
)

st.header("Try it yourself")
left, right = st.columns(2)
orders = left.slider("Orders before 1 September", 1, int(customers["orders"].max()), 5)
recency = right.slider(
    "Days since last order on 1 September", 1, int(customers["recency_days"].max()), 30
)
probability = predictor.probability(orders=orders, recency_days=recency)
st.metric("Estimated chance of returning by 9 December", f"{probability:.1%}")
st.caption("Holding the other input fixed, more orders raise the estimate and more days since the last order lower it.")

st.header("How much should we trust the estimate?")
st.markdown(
    "- The test customers came from the same 2011 cohort as the training customers. "
    "Results could change in a later season or with a different mix of customers.\n"
    "- The model estimates likely returns under historical conditions. A voucher trial "
    "would measure how much an offer changes that outcome.\n"
    "- **AI check.** An initial Codex calculation passed an unnamed row to the fitted model, "
    "which triggered a scikit-learn warning. The prediction method now builds a one-row DataFrame "
    "with the fitted feature names and order."
)
