
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


def main():
    input_path = Path("outputs/policy_eval/policy_eval_results.csv")
    out_dir = Path("paper_assets/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(input_path)

    # Use test split only.
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

    # Main trade-off: normal-operation false warning windows vs event detection rate.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for score_col, group in merged.groupby("score_col"):
        label = METHOD_NAMES.get(score_col, score_col)
        group = group.sort_values("threshold")
        ax.plot(
            group["normal_warning_windows"],
            group["event_detection_rate"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )

    # Highlight selected thresholds.
    for score_col, target_thr in SELECTED_THRESHOLDS.items():
        thr = nearest_threshold(merged, score_col, target_thr)
        if thr is None:
            continue

        row = merged[
            (merged["score_col"] == score_col)
            & (np.isclose(merged["threshold"], thr))
        ]

        if row.empty:
            continue

        row = row.iloc[0]
        ax.scatter(
            [row["normal_warning_windows"]],
            [row["event_detection_rate"]],
            s=90,
            marker="*",
            zorder=5,
        )
        ax.annotate(
            f'{METHOD_NAMES.get(score_col, score_col)}\nτ={thr:g}',
            xy=(row["normal_warning_windows"], row["event_detection_rate"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Warning windows during normal-operation days")
    ax.set_ylabel("Event detection rate on event-rich days")
    ax.set_title("Policy trade-off between event detection and normal-operation alarms")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()

    fig.savefig(out_dir / "policy_tradeoff_detection_vs_normal_windows.png", dpi=300)
    fig.savefig(out_dir / "policy_tradeoff_detection_vs_normal_windows.pdf")

    # Secondary figure: active minutes during normal operation.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for score_col, group in merged.groupby("score_col"):
        label = METHOD_NAMES.get(score_col, score_col)
        group = group.sort_values("threshold")
        ax.plot(
            group["active_minutes_per_normal_day"],
            group["event_detection_rate"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )

    for score_col, target_thr in SELECTED_THRESHOLDS.items():
        thr = nearest_threshold(merged, score_col, target_thr)
        if thr is None:
            continue

        row = merged[
            (merged["score_col"] == score_col)
            & (np.isclose(merged["threshold"], thr))
        ]

        if row.empty:
            continue

        row = row.iloc[0]
        ax.scatter(
            [row["active_minutes_per_normal_day"]],
            [row["event_detection_rate"]],
            s=90,
            marker="*",
            zorder=5,
        )
        ax.annotate(
            f'{METHOD_NAMES.get(score_col, score_col)}\nτ={thr:g}',
            xy=(row["active_minutes_per_normal_day"], row["event_detection_rate"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Active warning minutes per normal-operation day")
    ax.set_ylabel("Event detection rate on event-rich days")
    ax.set_title("Policy trade-off between detection and normal-operation burden")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()

    fig.savefig(out_dir / "policy_tradeoff_detection_vs_normal_active_minutes.png", dpi=300)
    fig.savefig(out_dir / "policy_tradeoff_detection_vs_normal_active_minutes.pdf")

    # Save the merged plotting table for paper audit.
    table_path = Path("paper_assets/tables/policy_tradeoff_plot_data.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(table_path, index=False)

    print("Saved figures:")
    print(out_dir / "policy_tradeoff_detection_vs_normal_windows.png")
    print(out_dir / "policy_tradeoff_detection_vs_normal_windows.pdf")
    print(out_dir / "policy_tradeoff_detection_vs_normal_active_minutes.png")
    print(out_dir / "policy_tradeoff_detection_vs_normal_active_minutes.pdf")
    print("Saved plot data:", table_path)

    selected_rows = []
    for score_col, target_thr in SELECTED_THRESHOLDS.items():
        thr = nearest_threshold(merged, score_col, target_thr)
        if thr is None:
            continue
        row = merged[
            (merged["score_col"] == score_col)
            & (np.isclose(merged["threshold"], thr))
        ].copy()
        if not row.empty:
            selected_rows.append(row)

    if selected_rows:
        selected = pd.concat(selected_rows, axis=0)
        cols = [
            "score_col",
            "threshold",
            "n_events",
            "event_detection_rate",
            "median_lead_minutes",
            "warning_precision",
            "false_windows_per_event_rich_day",
            "active_minutes_per_event_rich_day",
            "normal_warning_windows",
            "false_windows_per_normal_day",
            "active_minutes_per_normal_day",
        ]
        print("\nSelected operating points:")
        print(selected[cols].sort_values("score_col").to_string(index=False))


if __name__ == "__main__":
    main()
