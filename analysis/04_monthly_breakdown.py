"""월별 net_sales 합계 · 하루평균(월합계/그달 날짜수) 표, 그리고
월별 실제 매출 그래프 · 월별 7일 이동평균 그래프를 각각 월 단위로 분리해서 그린다."""
import sys

import matplotlib.pyplot as plt
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/daily_sales.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day

months = sorted(df["year_month"].unique())

# ------------------------------------------------------------------
# 월별 합계 / 하루평균(=합계/그달 날짜수) 표
# ------------------------------------------------------------------
monthly = (
    df.groupby("year_month")["net_sales"]
    .agg(days_in_data="size", net_sales_sum="sum", net_sales_daily_avg="mean")
    .reindex(months)
)
monthly["net_sales_sum"] = monthly["net_sales_sum"].round(2)
monthly["net_sales_daily_avg"] = monthly["net_sales_daily_avg"].round(2)

print(monthly.to_string())
monthly.to_csv(f"{TABLE_DIR}/monthly_net_sales_summary.csv", encoding="utf-8-sig")
print(f"\n저장 완료: {TABLE_DIR}/monthly_net_sales_summary.csv")


def month_grid(col, title, ylabel, color, fname):
    n = len(months)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=True)
    axes = axes.flatten()

    for i, ym in enumerate(months):
        ax = axes[i]
        sub = df.loc[df["year_month"] == ym].sort_values("day")
        ax.plot(sub["day"], sub[col], color=color, linewidth=1.5)
        ax.set_title(ym)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)

    for j in range(len(months), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("일(day)")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150)
    plt.close(fig)
    print(f"저장 완료: {FIG_DIR}/{fname}")


month_grid("net_sales", "월별 일별 순매출(net_sales)", "순매출", "#2E6F95",
           "monthly_net_sales_grid.png")
month_grid("net_sales_ma7", "월별 7일 후행 이동평균(net_sales_ma7)", "순매출(이동평균)", "#C1440E",
           "monthly_net_sales_ma7_grid.png")
