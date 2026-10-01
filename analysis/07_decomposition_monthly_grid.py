"""net_sales = trend(7일 중심이동평균) + weekday_effect(요일효과, 합 0) + resid(잔차) 로 분해.
실제값/트렌드/요일효과/잔차 4개를 각각 별도 그래프로, 월별 소그래프로 나눠 그린다."""
import sys

import matplotlib.pyplot as plt
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/daily_sales.csv"
OUT_PATH = "data/processed/daily_sales.csv"
FIG_DIR = "outputs/figures"

DOW_LABEL = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["day"] = df["date"].dt.day
df["dow_label"] = df["dow"].map(DOW_LABEL)

# 1) 추세: 7일 중심이동평균(당일 기준 앞뒤 3일, 총 7일)
df["trend"] = df["net_sales"].rolling(window=7, center=True, min_periods=7).mean()

# 2) 요일효과: resid1 = net_sales - trend 의 요일별 평균을, 합이 0이 되도록 정규화
resid1 = df["net_sales"] - df["trend"]
weekday_means = resid1.groupby(df["dow_label"]).mean()
weekday_effect_map = weekday_means - weekday_means.mean()
df["weekday_effect"] = df["dow_label"].map(weekday_effect_map)

# 3) 잔차
df["resid"] = df["net_sales"] - df["trend"] - df["weekday_effect"]

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("요일효과 (합 0):")
print(weekday_effect_map.reindex(DOW_ORDER).round(1).to_string())
print(f"\n분해식: net_sales = trend + weekday_effect + resid")
print(df[["date", "net_sales", "trend", "weekday_effect", "resid"]].head(10).to_string(index=False))

months = sorted(df["year_month"].unique())


def month_grid(col, title, ylabel, color, fname, sym_ylim=False):
    valid = df[col].dropna()
    n = len(months)
    ncols = 4
    nrows = -(-n // ncols)
    if sym_ylim:
        bound = valid.abs().max() * 1.1
        y_range = (-bound, bound)
    else:
        pad = (valid.max() - valid.min()) * 0.1
        y_range = (valid.min() - pad, valid.max() + pad)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=True)
    axes = axes.flatten()

    for i, ym in enumerate(months):
        ax = axes[i]
        sub = df.loc[df["year_month"] == ym].sort_values("day")
        ax.plot(sub["day"], sub[col], color=color, linewidth=1.5)
        if sym_ylim:
            ax.axhline(0, color="#8A8F98", linewidth=0.8)
        ax.set_title(ym)
        ax.set_ylim(*y_range)
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


month_grid("net_sales", "월별 실제 순매출(net_sales)", "순매출", "#2E6F95",
           "decomp_actual_monthly_grid.png")
month_grid("trend", "월별 추세(7일 중심이동평균)", "순매출", "#2E6F95",
           "decomp_trend_monthly_grid.png")
month_grid("weekday_effect", "월별 요일효과(weekday_effect)", "순매출 차이", "#C1440E",
           "decomp_weekday_effect_monthly_grid.png", sym_ylim=True)
month_grid("resid", "월별 잔차(resid)", "순매출 차이", "#8A8F98",
           "decomp_resid_monthly_grid.png", sym_ylim=True)
