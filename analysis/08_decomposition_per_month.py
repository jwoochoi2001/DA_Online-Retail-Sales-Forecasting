"""net_sales 분해(실제값/추세/요일효과/잔차) 4개 그래프를 달마다 따로(파일 1개=1개월, 4단 subplot) 그린다."""
import sys

import matplotlib.pyplot as plt
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/daily_sales.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["day"] = df["date"].dt.day

months = sorted(df["year_month"].unique())

y_actual = (df["net_sales"].min(), df["net_sales"].max() * 1.05)
y_trend = (df["trend"].min(), df["trend"].max() * 1.05)
bound_wd = df["weekday_effect"].abs().max() * 1.1
bound_resid = df["resid"].abs().max() * 1.1

panels = [
    ("net_sales", "실제값(net_sales)", "#2E6F95", y_actual, False),
    ("trend", "추세(7일 중심이동평균)", "#2E6F95", y_trend, False),
    ("weekday_effect", "요일효과", "#C1440E", (-bound_wd, bound_wd), True),
    ("resid", "잔차", "#8A8F98", (-bound_resid, bound_resid), True),
]

for ym in months:
    sub = df.loc[df["year_month"] == ym].sort_values("day")

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for ax, (col, label, color, ylim, zero_line) in zip(axes, panels):
        ax.plot(sub["day"], sub[col], color=color, linewidth=1.6)
        if zero_line:
            ax.axhline(0, color="#8A8F98", linewidth=0.8)
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("일(day)")
    fig.suptitle(f"{ym} 순매출 분해: net_sales = trend + weekday_effect + resid", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = f"decomp_{ym}.png"
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150)
    plt.close(fig)
    print(f"저장 완료: {FIG_DIR}/{fname}")
