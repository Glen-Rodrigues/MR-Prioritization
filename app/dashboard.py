"""MR-Rank dashboard (Phase 10).

Run from the repo root:  streamlit run app/dashboard.py
Reads only saved files in outputs/, so no torch or model is needed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"

st.set_page_config(page_title="MR-Rank", layout="wide")


@st.cache_data
def load():
    meta = json.load(open(OUT / "kill_matrix_metadata.json"))
    km = np.load(OUT / "kill_matrix.npy").astype(bool)  # shape (MRs, mutants)
    greedy = pd.DataFrame(json.load(open(OUT / "greedy_ranking.json")))
    model_b = json.load(open(OUT / "model_b_predicted_ranking.json"))
    ev = json.load(open(OUT / "evaluation_metrics.json"))
    return meta, km, greedy, model_b, ev


meta, km, greedy, model_b, ev = load()
mr_ids, mutant_ids = meta["mr_ids"], meta["mutant_ids"]
mr_idx = {m: i for i, m in enumerate(mr_ids)}
greedy_order = greedy["mr_id"].tolist()
detectable = km.any(axis=0)
n_det = int(detectable.sum())


def covered(order, k):
    rows = [mr_idx[m] for m in order[:k]]
    return int(km[rows].any(axis=0).sum())


kill_sets = {m: set(np.where(km[i])[0].tolist()) for m, i in mr_idx.items()}
norm_cost = dict(zip(greedy["mr_id"], greedy["norm_cost"]))


def _jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def greedy_rank(alpha, beta, gamma):
    """Same logic as ranking_module.greedy_rank, with adjustable weights."""
    remaining, selected, out = list(mr_ids), [], []
    while remaining:
        scores, divs = {}, {}
        for mr in remaining:
            divs[mr] = 1.0 if not selected else 1 - max(_jaccard(kill_sets[mr], kill_sets[s]) for s in selected)
            scores[mr] = alpha * meta["fdr_per_mr"][mr] + beta * (1 - norm_cost[mr]) + gamma * divs[mr]
        best = max(scores, key=scores.get)
        out.append({"rank": len(out) + 1, "mr_id": best, "priority_score": scores[best],
                    "fdr": meta["fdr_per_mr"][best], "diversity": divs[best]})
        selected.append(best)
        remaining.remove(best)
    return out


st.title("MR-Rank: Metamorphic Relation Prioritization")
st.caption("System under test: ResNet-18 on CIFAR-10  |  20 MRs x 80 mutants")

tab_over, tab_rank, tab_km, tab_eval, tab_ab = st.tabs(
    ["Overview", "MR Ranking", "Kill Matrix", "Evaluation", "Model A vs B"]
)

# ---------------------------------------------------------------- Overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metamorphic relations", len(mr_ids))
    c2.metric("Mutants", len(mutant_ids))
    c3.metric("Detectable mutants", n_det, help="Killed by at least one MR")
    det = ev.get("apfd_detectable_only")
    if det:
        c4.metric("APFD (detectable)", f"{det['greedy']:.3f}",
                  delta=f"{det['greedy'] - det['random_mean']:+.3f} vs random")
    st.markdown(
        "**Pipeline:** Model A / Model B -> 20 MR transforms -> 80 mutants -> "
        "20x80 Kill Matrix -> Greedy ranking (Model A) and Meta-Classifier "
        "ranking (Model B) -> APFD / FD@k evaluation."
    )
    st.markdown(
        f"**Mutants not detected by any MR: {len(mutant_ids) - n_det}** "
        "(mostly deep-layer noise/zeroing mutants that leave the model unchanged)."
    )

# ----------------------------------------------------------------- Ranking
with tab_rank:
    st.subheader("Greedy ranking (Model A)")
    k = st.slider("MR budget k (run only the top-k MRs)", 1, len(mr_ids), 3)
    found = covered(greedy_order, k)
    rand = ev["fd_at_k"]["random"][k - 1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Mutants detected", f"{found} / {n_det}")
    m2.metric("Coverage", f"{found / n_det:.1%}")
    m3.metric("Random baseline", f"{rand:.1%}")

    df = greedy.copy()
    df["selected"] = df["rank"] <= k
    fig = px.bar(df, x="mr_id", y="priority_score", color="selected",
                 category_orders={"mr_id": greedy_order},
                 color_discrete_map={True: "#1f4e79", False: "#c8c8c8"},
                 hover_data=["fdr", "norm_cost", "diversity"])
    fig.update_layout(xaxis_title="", yaxis_title="Priority score", legend_title="In top-k")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(greedy.set_index("rank"), width="stretch")

    st.subheader("Try different weights")
    st.caption("Priority = alpha*FDR + beta*(1 - NormCost) + gamma*Diversity. Paper defaults: 0.5 / 0.2 / 0.3.")
    w1, w2, w3 = st.columns(3)
    alpha = w1.slider("alpha (FDR)", 0.0, 1.0, 0.5, 0.05)
    beta = w2.slider("beta (cost)", 0.0, 1.0, 0.2, 0.05)
    gamma = w3.slider("gamma (diversity)", 0.0, 1.0, 0.3, 0.05)
    custom = greedy_rank(alpha, beta, gamma)
    custom_order = [r["mr_id"] for r in custom]
    cmp_cols = st.columns(2)
    cmp_cols[0].metric(f"Coverage at k={k} (custom weights)", f"{covered(custom_order, k) / n_det:.1%}")
    cmp_cols[1].metric(f"Coverage at k={k} (default weights)", f"{found / n_det:.1%}")
    if custom_order == greedy_order:
        st.success("Same ordering as the saved Greedy ranking.")
    else:
        first_diff = next(i for i, (x, y) in enumerate(zip(custom_order, greedy_order)) if x != y)
        st.warning(f"Ordering differs from the default starting at rank {first_diff + 1}.")
    st.dataframe(pd.DataFrame(custom).set_index("rank").round(4), width="stretch")

# -------------------------------------------------------------- Kill Matrix
with tab_km:
    st.subheader("Kill Matrix (rows = MRs, columns = mutants)")
    by_rank = st.checkbox("Order rows by Greedy rank", value=True)
    rows = greedy_order if by_rank else mr_ids
    z = km[[mr_idx[m] for m in rows]].astype(int)
    heat = go.Figure(go.Heatmap(z=z, x=mutant_ids, y=rows, showscale=False,
                                colorscale=[[0, "#f1f1f1"], [1, "#1f4e79"]]))
    heat.update_yaxes(autorange="reversed")
    heat.update_xaxes(showticklabels=False)
    heat.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, width="stretch")

    left, right = st.columns(2)
    fdr = pd.Series(meta["fdr_per_mr"]).sort_values(ascending=False)
    with left:
        st.markdown("**Fault detection rate per MR**")
        st.plotly_chart(px.bar(fdr, labels={"index": "", "value": "FDR"}).update_layout(showlegend=False),
                        width="stretch")
    with right:
        st.markdown("**Undetected mutants (by group)**")
        undetected = [m for m, d in zip(mutant_ids, detectable) if not d]
        groups = pd.Series([m.rsplit("_run", 1)[0] for m in undetected]).value_counts()
        st.dataframe(groups.rename("count"), width="stretch")

# -------------------------------------------------------------- Evaluation
with tab_eval:
    st.subheader("FD@k: fraction of detectable mutants found by the top-k MRs")
    fk = pd.DataFrame({"k": range(1, len(mr_ids) + 1),
                       "Greedy": ev["fd_at_k"]["greedy"],
                       "FDR-only": ev["fd_at_k"]["fdr_only"],
                       "Random (mean)": ev["fd_at_k"]["random"]})
    st.plotly_chart(px.line(fk.melt("k", var_name="Ordering", value_name="FD@k"),
                            x="k", y="FD@k", color="Ordering", markers=True),
                    width="stretch")

    st.subheader("APFD")

    def apfd(order, cols):
        sub = km[:, cols]
        n, m = len(order), sub.shape[1]
        tf = 0
        for j in range(m):
            hits = [r for r, mr in enumerate(order, 1) if sub[mr_idx[mr], j]]
            tf += hits[0] if hits else n
        return 1 - tf / (n * m) + 1 / (2 * n)

    all_cols, det_cols = np.ones(km.shape[1], dtype=bool), detectable
    orders = {
        "FDR-only": sorted(mr_ids, key=lambda m: -meta["fdr_per_mr"][m]),
        "Greedy (Model A)": greedy_order,
        "Meta-Classifier (Model B)": [r["mr_id"] for r in model_b["predicted_ranking"]],
    }
    rand_orders = []
    for s in range(30):
        o = list(mr_ids)
        np.random.RandomState(s).shuffle(o)
        rand_orders.append(o)
    rand_all = [apfd(o, all_cols) for o in rand_orders]
    rand_det = [apfd(o, det_cols) for o in rand_orders]

    table = pd.DataFrame({
        "Ordering": ["Random (mean of 30)"] + list(orders),
        "APFD (all 80)": [np.mean(rand_all)] + [apfd(o, all_cols) for o in orders.values()],
        f"APFD (detectable {n_det})": [np.mean(rand_det)] + [apfd(o, det_cols) for o in orders.values()],
    })
    st.dataframe(table.set_index("Ordering").round(4), width="stretch")

    g_all = apfd(greedy_order, all_cols)
    wins = sum(r < g_all for r in rand_all)
    pval = ev.get("wilcoxon", {}).get("p_value")
    st.write(f"Greedy beats **{wins} of {len(rand_all)}** random orderings"
             + (f" (Wilcoxon p = {pval:.2g})." if pval is not None else "."))
    st.info(
        "Limitations: (1) MR15 and MR12 alone cover every detectable mutant, so APFD cannot "
        "separate Greedy from FDR-only. (2) Rankings are evaluated on the same Kill Matrix "
        "they were built from (in-sample). (3) The Wilcoxon test compares one deterministic "
        "value against 30 random orderings, so treat it as a simplification."
    )

# --------------------------------------------------------------- Model A/B
with tab_ab:
    st.subheader("Model A (Greedy) vs Model B (Meta-Classifier) ranking")
    b = pd.DataFrame(model_b["predicted_ranking"]).rename(columns={"rank": "rank_B"})
    a_ = greedy[["mr_id", "rank"]].rename(columns={"rank": "rank_A"})
    cmp_ = a_.merge(b, on="mr_id")
    cmp_["moved_up_in_B"] = cmp_["rank_A"] - cmp_["rank_B"]

    rho = cmp_[["rank_A", "rank_B"]].corr(method="spearman").iloc[0, 1]
    st.metric("Spearman rank correlation", f"{rho:.2f}")

    slope = go.Figure()
    for _, r in cmp_.iterrows():
        colour = "#1f4e79" if r["moved_up_in_B"] > 0 else "#b5651d" if r["moved_up_in_B"] < 0 else "#999999"
        slope.add_trace(go.Scatter(x=["Model A", "Model B"], y=[r["rank_A"], r["rank_B"]],
                                   mode="lines+markers+text", line=dict(color=colour),
                                   text=[r["mr_id"], r["mr_id"]], textposition=["middle left", "middle right"],
                                   showlegend=False, hoverinfo="text+y"))
    slope.update_yaxes(autorange="reversed", title="Rank")
    slope.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(slope, width="stretch")

    st.dataframe(cmp_.reindex(cmp_["moved_up_in_B"].abs().sort_values(ascending=False).index)
                 .set_index("mr_id").round(3), width="stretch")
    st.caption("Model B's ranking is scored against Model A's Kill Matrix, so this is a proxy "
               "comparison rather than independent validation.")
