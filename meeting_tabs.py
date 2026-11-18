import ast
import os
import re
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, HTML

def _read_current_rate(script_path, var_name, fallback=0.0):
    if not os.path.exists(script_path):
        return fallback
    with open(script_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=script_path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                        return float(node.value.value)
    return fallback

def _short_tab_label(meeting_name):
    m = re.search(r"([A-Za-z]{3})\s+(\d{1,2})", str(meeting_name))
    if m:
        month, day = m.group(1), int(m.group(2))
        return f"{day}/{month}"
    return str(meeting_name).split(":")[-1].strip()

def _parse_numeric_level(label):
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(label))]
    if len(nums) >= 2:
        return sum(nums[:2]) / 2.0
    if len(nums) == 1:
        return nums[0]
    return 0.0

def render_meeting_tabs(prob_csv, engine_script, rate_var, max_meetings=8):
    if not os.path.exists(prob_csv):
        print(f"Probability file not found: {prob_csv}")
        return

    df = pd.read_csv(prob_csv)
    if df.empty:
        return

    cur_rate = _read_current_rate(engine_script, rate_var)
    x_col = "target_bracket" if "target_bracket" in df.columns else "target_rate"

    # Extract numeric sort order directly from the label string (works for all 7 banks)
    unique_labels = df[x_col].drop_duplicates().tolist()
    level_map = {lbl: _parse_numeric_level(lbl) for lbl in unique_labels}
    ordered_labels = sorted(unique_labels, key=lambda lbl: level_map[lbl])

    meetings = df["meeting_name"].drop_duplicates().tolist()[:max_meetings]

    print("::: {.panel-tabset}\n")
    for mtg in meetings:
        tab_title = _short_tab_label(mtg)
        print(f"#### {tab_title}\n")

        sub = df[df["meeting_name"] == mtg].set_index(x_col)
        probs = [float(sub.loc[lbl, "probability"]) if lbl in sub.index else 0.0 for lbl in ordered_labels]
        texts = [f"{p:.1f}%" if p >= 0.5 else "" for p in probs]

        colors = []
        for lbl in ordered_labels:
            val = level_map[lbl]
            if val < cur_rate - 0.05:
                colors.append("#2b6cb0")   # Cut
            elif val > cur_rate + 0.05:
                colors.append("#f6d58e")   # Hike
            else:
                colors.append("#718096")   # Hold

        fig = go.Figure(
            go.Bar(
                x=ordered_labels,
                y=probs,
                marker_color=colors,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>", showlegend=False,
            )
        )

        for label, color in [("Cut", "#2b6cb0"), ("Hold", "#718096"), ("Hike", "#f6d58e")]:
            fig.add_trace(
                go.Bar(
                    x=[None], y=[None], name=label,
                    marker_color=color, showlegend=True
                )
            )

        fig.update_layout(
            barmode="overlay",
            height=360,
            margin=dict(l=40, r=20, t=35, b=45),
            yaxis=dict(
                title="Probability",
                range=[0, 112],
                ticksuffix="%",
                gridcolor="rgba(128,128,128,0.2)"
            ),
            xaxis=dict(title="", type="category"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
        )

        display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
        print("\n")

    print(":::\n")
