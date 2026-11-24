"""Group 5 - Do cheaper products sell more?"""
import matplotlib.pyplot as plt
import streamlit as st

from ml_base import load_products
from w03_group5_price_demand import PriceDemand          # <- OUR module and OUR class

st.title("🏷️ Do cheaper products sell more?")
st.caption("Card 5 · Team: Max, Yehor, Natan, Jan")

products = load_products()
model = PriceDemand(products).fit()
r = model.results_

# the same class, but "units" replaced by the number of ORDERS (invoices) of each product
orders_model = PriceDemand(products.assign(units=products["invoices"])).fit()

st.header("1️⃣ What we built")
st.markdown(f"""
`PriceDemand(products)` - a **log-log linear regression** of the units sold on the price, over **{r['n_products']:,} products**
(Dec 2010 - Dec 2011). On a log scale the slope is the **price elasticity**: −0.5 = "10% cheaper → about 5% more units".

`.fit()` → elasticity, intercept, R² · `.predict_units(price)` → units a typical product at that price sells in a year ·
`.search(word)` → find products · `.plot()` → the chart below
""")

st.header("2️⃣ Our answer to the board")
units_change = 0.9 ** r["elasticity"] - 1               # price × 0.9  ->  units × 0.9^elasticity
revenue_change = 0.9 ** (1 + r["elasticity"]) - 1       # revenue = price × units
col1, col2, col3, col4 = st.columns(4)
col1.metric("Price elasticity", r["elasticity"])
col2.metric("R²", r["r2"], help="share of the differences in units that price explains")
col3.metric("10% price cut → units", f"{units_change:+.1%}")
col4.metric("10% price cut → revenue", f"{revenue_change:+.1%}")

ax = model.plot()
st.pyplot(ax.figure)
plt.close(ax.figure)

st.markdown(f"""
**No - the regression does not support a 10% price cut.**
- Cheaper products do sell more **units**, but only a little: 10% cheaper → about **{units_change:+.1%}** units, so the
  **revenue falls by {-revenue_change:.1%}** (elasticity {r['elasticity']} is between −1 and 0 = demand is *inelastic*).
- Cheap products are not bought **more often**: the same class on the number of **orders** gives an elasticity of
  only **{orders_model.results_['elasticity']}** (R² {orders_model.results_['r2']}). They are bought in **bigger packs** -
  among the 10 best sellers, gliders go 48 at a time, paint sets 36, cake cases 24 (`median_qty` = pieces per order line). The "extra units" are a pack-size effect, not extra customers.
- **R² = {r['r2']}**: price explains only {r['r2']:.0%} of the differences between products. Design, season and the type of product matter far more.
""")

st.header("🎛️ Try it yourself")
col_a, col_b = st.columns(2)
price = col_a.number_input("Price of one piece (GBP)", min_value=0.05, max_value=150.0, value=2.50, step=0.25)
col_b.metric("A typical product at this price sells", f"{model.predict_units(price):,} units / year",
             delta=f"{model.predict_units(price * 0.9) - model.predict_units(price):+,} if 10% cheaper")

word = st.text_input("🔎 Search a product (e.g. heart, bag, cake, christmas)", "heart")
if word.strip():
    found = model.search(word)
    st.write(f"**{len(found)}** products found - the real units next to what our model expects for their price:")
    st.dataframe(found.head(50))

st.header("3️⃣ Is our number real? + what the AI got wrong")
st.markdown("""
- **Different products, not different prices:** we compare a 50p pack of cake cases with a £15 product. The data never show the
  **same** product at two prices, so the elasticity is a correlation, not proof of what a price cut would do.
- **Pack sizes:** cheap items are sold by the dozen to wholesale customers - that explains much of the slope (see the orders elasticity above).
- **Only products with ≥ 10 invoices**: products that flopped are not in the table (survivorship).
- **AI:** Claude wrote the product table with `st.dataframe(..., use_container_width=True)` - Streamlit warned that this
  argument is deprecated and will be removed, so the page would break online later. We removed it. We also did not trust the
  elasticity until we checked it by hand: `np.polyfit` on the logs gives the same −0.63.
""")
