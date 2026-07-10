
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METHOD_NAMES = {
    "co_relative": "Current relative CO",
    "rf_score": "Random forest risk score",
    "hgnn_score": "HGNN-risk score",
    "fusion_score": "CO-gated HGNN fusion",
    "risk_target": "Oracle future risk",
}

METHOD_ABBR = {
    "co_relative": "CO",
    "rf_score": "RF",
    "hgnn_score": "HGNN",
    "fusion_score": "Fusion",
    "risk_target": "Oracle",
}

METHOD_ORDER = [
    "co_relative",
    "rf_score",
    "hgnn_score",
    "fusion_score",
    "risk_target",
]

SELECTED_THRESHOLDS = {
    "co_relative": 0.05,
    "rf_score": 0.30,
    "hgnn_score": 0.20,
    "fusion_score": 0.016,
    "risk_target": 0.20,
}


def nearest_threshold(df, score_col, target):
    sub = df[df["score_col"] == score_col].copy()
    if sub.empty:
        return None
    sub["dist"] = (sub["threshold"] - target).abs()
    return float(sub.sort_values("dist").iloc[0]["threshold"])


def build_merged(res):
    test = res[res["split"] == "test"].copy()
    event = test[test["period_type"] == "event_rich"].copy()
    normal = test[test["period_type"] == "normal_operation"].copy()

    key = ["score_col", "threshold"]

    merged = event.merge(
        normal[
            key
            + [
                "n_warning_windows",
                "false_windows_per_day",
                "active_minutes_per_day",
            ]
        ],
        on=key,
        how="left",
        suffixes=("_event", "_normal"),
    )

    merged = merged.rename(
        columns={
            "n_warning_windows_event": "event_warning_windows",
            "false_windows_per_day_event": "false_windows_per_event_rich_day",
            "active_minutes_per_day_event": "active_minutes_per_event_rich_day",
            "n_warning_windows_normal": "normal_warning_windows",
            "false_windows_per_day_normal": "false_windows_per_normal_day",
            "active_minutes_per_day_normal": "active_minutes_per_normal_day",
        }
    )

    return merged


def selected_points(merged):
    rows = []
    for score_col, target_thr in SELECTED_THRESHOLDS.items():
        thr = nearest_threshold(merged, score_col, target_thr)
        if thr is None:
            continue

        row = merged[
            (merged["score_col"] == score_col)
            & np.isclose(merged["threshold"], thr)
        ].copy()

        if not row.empty:
            row["selected_threshold"] = thr
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, axis=0)


def plot_selected_operating_points(selected, out_dir):
    selected = selected.copy()
    selected["method_name"] = selected["score_col"].map(METHOD_NAMES)
    selected["abbr"] = selected["score_col"].map(METHOD_ABBR)

    # exclude oracle from the deployable-method plot, then add oracle separately as upper bound
    deploy = selected[selected["score_col"] != "risk_target"].copy()
    oracle = selected[selected["score_col"] == "risk_target"].copy()

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for score_col in METHOD_ORDER:
        row = selected[selected["score_col"] == score_col]
        if row.empty:
            continue

        row = row.iloc[0]
        marker = "*" if score_col == "fusion_score" else "o"
        size = 190 if score_col == "fusion_score" else 95
        alpha = 1.0 if score_col != "risk_target" else 0.6

        ax.scatter(
            row["normal_warning_windows"],
            row["event_detection_rate"],
            s=size,
            marker=marker,
            alpha=alpha,
            label=f"{METHOD_NAMES[score_col]} (τ={row['threshold']:g})",
        )

        # manual offsets to avoid overlap
        offsets = {
            "co_relative": (8, 8),
            "rf_score": (8, -18),
            "hgnn_score": (8, 8),
            "fusion_score": (8, 10),
            "risk_target": (8, -18),
        }
        dx, dy = offsets.get(score_col, (8, 8))

        ax.annotate(
            METHOD_ABBR[score_col],
            xy=(row["normal_warning_windows"], row["event_detection_rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Warning windows during normal-operation days")
    ax.set_ylabel("Event detection rate on event-rich days")
    ax.set_title("Selected operating points under event-level warning policy")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.90)
    ax.legend(fontsize=8, frameon=True, loc="lower right")
    fig.tight_layout()

    fig.savefig(out_dir / "selected_policy_operating_points.png", dpi=300)
    fig.savefig(out_dir / "selected_policy_operating_points.pdf")
    plt.close(fig)


def plot_clean_curves(merged, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for score_col in METHOD_ORDER:
        group = merged[merged["score_col"] == score_col].copy()
        if group.empty:
            continue

        group = group.sort_values("threshold")
        ax.plot(
            group["normal_warning_windows"],
            group["event_detection_rate"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=METHOD_NAMES[score_col],
        )

        thr = nearest_threshold(merged, score_col, SELECTED_THRESHOLDS[score_col])
        row = group[np.isclose(group["threshold"], thr)]
        if not row.empty:
            row = row.iloc[0]
            ax.scatter(
                [row["normal_warning_windows"]],
                [row["event_detection_rate"]],
                s=120,
                marker="*",
                zorder=5,
            )

    ax.set_xlabel("Warning windows during normal-operation days")
    ax.set_ylabel("Event detection rate on event-rich days")
    ax.set_title("Threshold trade-off curves")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()

    fig.savefig(out_dir / "policy_tradeoff_curves_clean.png", dpi=300)
    fig.savefig(out_dir / "policy_tradeoff_curves_clean.pdf")
    plt.close(fig)


def main():
    res = pd.read_csv("outputs/policy_eval/policy_eval_results.csv")
    out_dir = Path("paper_assets/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = build_merged(res)
    selected = selected_points(merged)

    table_dir = Path("paper_assets/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    selected_cols = [
        "score_col",
        "threshold",
        "n_events",
        "event_warning_windows",
        "event_detection_rate",
        "median_lead_minutes",
        "warning_precision",
        "false_windows_per_event_rich_day",
        "active_minutes_per_event_rich_day",
        "normal_warning_windows",
        "false_windows_per_normal_day",
        "active_minutes_per_normal_day",
    ]

    selected[selected_cols].to_csv(
        table_dir / "selected_policy_operating_points.csv",
        index=False,
    )

    plot_selected_operating_points(selected, out_dir)
    plot_clean_curves(merged, out_dir)

    print("Saved clean figures:")
    print(out_dir / "selected_policy_operating_points.png")
    print(out_dir / "selected_policy_operating_points.pdf")
    print(out_dir / "policy_tradeoff_curves_clean.png")
    print(out_dir / "policy_tradeoff_curves_clean.pdf")
    print()
    print("Selected operating points:")
    print(selected[selected_cols].sort_values("score_col").to_string(index=False))


if __name__ == "__main__":
    main()
