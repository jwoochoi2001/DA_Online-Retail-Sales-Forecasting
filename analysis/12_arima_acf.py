"""ARIMA 비교를 위한 사전 점검: 학습기간(2011-01-01~2011-10-31) 일별 판매금액 그래프와
시차 1~21일 자기상관(ACF) 그래프. 7/14/21일 전 값과의 상관계수는 따로 표시한다."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/daily_sales.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df.index.freq = "D"

train = df.loc["2011-01-01":"2011-10-31", "net_sales"]
print(f"학습: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")

# ------------------------------------------------------------------
# 그래프 1: 학습기간 일별 판매금액
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train.index, train.values, color="#2E6F95", linewidth=1.2)
ax.set_title("학습기간 일별 판매금액(net_sales), 2011-01-01 ~ 2011-10-31")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/arima_train_daily_sales.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 자기상관(ACF), 시차 1~21일
# ------------------------------------------------------------------
n_lags = 21
acf_vals, confint = acf(train, nlags=n_lags, alpha=0.05, fft=True)
# acf_vals[0]는 시차 0(자기 자신, 항상 1) -> 1~21만 사용
lags = np.arange(1, n_lags + 1)
acf_1_21 = acf_vals[1:n_lags + 1]
conf_1_21 = confint[1:n_lags + 1] - acf_vals[1:n_lags + 1, None]  # 대칭 오차폭

highlight_lags = {7, 14, 21}
colors = ["#C1440E" if lag in highlight_lags else "#2E6F95" for lag in lags]

fig2, ax2 = plt.subplots(figsize=(12, 5.5))
ax2.bar(lags, acf_1_21, color=colors, width=0.6)
ax2.fill_between(lags, conf_1_21[:, 0], conf_1_21[:, 1], color="#D9D9D9", alpha=0.5, step=None)
ax2.axhline(0, color="#8A8F98", linewidth=0.8)
ax2.set_title("일별 판매금액 자기상관(ACF), 시차 1~21일 (학습기간)")
ax2.set_xlabel("시차(lag, 일)")
ax2.set_ylabel("자기상관계수")
ax2.set_xticks(lags)
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

for lag in highlight_lags:
    val = acf_1_21[lag - 1]
    ax2.annotate(f"{val:.3f}", (lag, val), textcoords="offset points",
                 xytext=(0, 8 if val >= 0 else -14), ha="center", fontsize=9, color="#C1440E", fontweight="bold")

fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/arima_acf_1_21.png", dpi=150)
plt.close(fig2)

# ------------------------------------------------------------------
# 7/14/21일 전 값과의 상관계수 별도 표시
# ------------------------------------------------------------------
highlight_df = pd.DataFrame({
    "lag(일)": sorted(highlight_lags),
    "자기상관계수": [acf_1_21[lag - 1] for lag in sorted(highlight_lags)],
})
print("\n=== 7/14/21일 전 값과의 자기상관계수 ===")
print(highlight_df.round(4).to_string(index=False))

acf_table = pd.DataFrame({"lag": lags, "acf": acf_1_21})
acf_table.to_csv(f"{TABLE_DIR}/arima_acf_1_21.csv", index=False, encoding="utf-8-sig")
highlight_df.to_csv(f"{TABLE_DIR}/arima_acf_7_14_21.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {FIG_DIR}/arima_train_daily_sales.png")
print(f"저장 완료: {FIG_DIR}/arima_acf_1_21.png")
print(f"저장 완료: {TABLE_DIR}/arima_acf_1_21.csv")
print(f"저장 완료: {TABLE_DIR}/arima_acf_7_14_21.csv")
