"""일별 순매출(net_sales) 추이 선그래프. 휴무일(is_closed)과 이상치(is_outlier)를 표시."""
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

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(df["date"], df["net_sales"], color="#2E6F95", linewidth=1.2, label="일별 순매출(net_sales)")

closed = df.loc[df["is_closed"] == 1]
ax.scatter(closed["date"], closed["net_sales"], color="#8A8F98", s=14, zorder=3, label="휴무일(is_closed)")

outlier = df.loc[df["is_outlier"] == 1]
ax.scatter(outlier["date"], outlier["net_sales"], color="#C1440E", s=45, zorder=4, marker="D",
           label="이상치(is_outlier)")

ax.set_title("온라인 소매 일별 순매출 추이 (2010-12-01 ~ 2011-12-08)")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/daily_net_sales_trend.png", dpi=150)
plt.close(fig)

print(f"저장 완료: {FIG_DIR}/daily_net_sales_trend.png")
