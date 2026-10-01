"""전체 기간(2010-12-01~2011-12-08) 요일별 net_sales 통계표와 그래프.
휴무일이 요일마다 다르게 끼어 있어(토요일은 거의 전부 휴무) 전체평균과 영업일평균을 함께 낸다."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/daily_sales.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

DOW_LABEL = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df["dow_label"] = df["dow"].map(DOW_LABEL)

open_df = df.loc[df["is_closed"] == 0]

summary = pd.DataFrame(index=DOW_ORDER)
summary["n_days"] = df.groupby("dow_label").size().reindex(DOW_ORDER)
summary["n_closed"] = df.loc[df["is_closed"] == 1].groupby("dow_label").size().reindex(DOW_ORDER).fillna(0).astype(int)
summary["n_open"] = summary["n_days"] - summary["n_closed"]
summary["avg_all"] = df.groupby("dow_label")["net_sales"].mean().reindex(DOW_ORDER)
summary["avg_open"] = open_df.groupby("dow_label")["net_sales"].mean().reindex(DOW_ORDER)
summary["median_open"] = open_df.groupby("dow_label")["net_sales"].median().reindex(DOW_ORDER)
summary["std_open"] = open_df.groupby("dow_label")["net_sales"].std().reindex(DOW_ORDER)
summary["sum_all"] = df.groupby("dow_label")["net_sales"].sum().reindex(DOW_ORDER)
summary["share_%"] = (summary["sum_all"] / summary["sum_all"].sum() * 100)

summary = summary.round(1)
print(summary.to_string())

summary.to_csv(f"{TABLE_DIR}/weekday_net_sales_summary.csv", encoding="utf-8-sig")
print(f"\n저장 완료: {TABLE_DIR}/weekday_net_sales_summary.csv")

# ------------------------------------------------------------------
# 그래프 1: 요일별 전체평균 vs 영업일평균 막대
# ------------------------------------------------------------------
x = np.arange(len(DOW_ORDER))
width = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width / 2, summary["avg_all"], width, color="#8A8F98", label="전체평균(휴무일 0 포함)")
b2 = ax.bar(x + width / 2, summary["avg_open"], width, color="#2E6F95", label="영업일평균(휴무일 제외)")
ax.set_title("요일별 평균 순매출: 전체평균 vs 영업일평균")
ax.set_xlabel("요일")
ax.set_ylabel("평균 순매출")
ax.set_xticks(x)
ax.set_xticklabels(DOW_ORDER)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
for bars in (b1, b2):
    for bar in bars:
        val = bar.get_height()
        if val > 0:
            ax.annotate(f"{val:,.0f}", (bar.get_x() + bar.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/weekday_avg_all_vs_open.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 그래프 2: 요일별(영업일만) net_sales 분포 boxplot
# ------------------------------------------------------------------
box_data = [open_df.loc[open_df["dow_label"] == d, "net_sales"].values for d in DOW_ORDER]
fig2, ax2 = plt.subplots(figsize=(9, 5))
bp = ax2.boxplot(box_data, tick_labels=DOW_ORDER, patch_artist=True, showmeans=True,
                  medianprops=dict(color="#C1440E", linewidth=1.5),
                  meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#C1440E", markersize=5))
for patch in bp["boxes"]:
    patch.set_facecolor("#DCE8EF")
    patch.set_edgecolor("#2E6F95")
ax2.set_title("요일별 순매출 분포 (영업일만, 상자그림)")
ax2.set_xlabel("요일")
ax2.set_ylabel("순매출")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/weekday_net_sales_boxplot.png", dpi=150)
plt.close(fig2)

print(f"저장 완료: {FIG_DIR}/weekday_avg_all_vs_open.png")
print(f"저장 완료: {FIG_DIR}/weekday_net_sales_boxplot.png")
