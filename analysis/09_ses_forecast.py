"""단순지수평활법(SES)으로 net_sales를 학습(2011-01-01~2011-10-31)하고 11월(2011-11-01~2011-11-30)을 예측한다.
평활계수(alpha)와 초기수준(level)은 학습 자료로 자동 추정한다(statsmodels optimized=True)."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

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
test = df.loc["2011-11-01":"2011-11-30", "net_sales"]

print(f"학습: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")
print(f"검증: {test.index.min().date()} ~ {test.index.max().date()} ({len(test)}일)")

# 평활계수(alpha)와 초기수준을 학습 자료로 자동 추정(SSE 최소화)
model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
alpha = model.params["smoothing_level"]
initial_level = model.params["initial_level"]

forecast = model.forecast(len(test))
forecast.index = test.index

print(f"\n평활계수(alpha) = {alpha:.4f}")
print(f"초기수준(initial_level) = {initial_level:,.2f}")
print(f"SES 예측값(11월, 모든 날짜 동일) = {forecast.iloc[0]:,.2f}")

mae = (test - forecast).abs().mean()
rmse = np.sqrt(((test - forecast) ** 2).mean())
mape_mask = test != 0
mape = ((test[mape_mask] - forecast[mape_mask]).abs() / test[mape_mask]).mean() * 100

print(f"\nMAE  = {mae:,.2f}")
print(f"RMSE = {rmse:,.2f}")
print(f"MAPE = {mape:.2f}% (net_sales=0인 휴무일 {(~mape_mask).sum()}일은 분모 0이라 제외)")

metrics = pd.DataFrame({
    "alpha": [alpha], "initial_level": [initial_level],
    "MAE": [mae], "RMSE": [rmse], "MAPE_%": [mape],
})
metrics.to_csv(f"{TABLE_DIR}/ses_forecast_metrics.csv", index=False, encoding="utf-8-sig")

result = pd.DataFrame({"actual": test, "forecast": forecast})
result.to_csv(f"{TABLE_DIR}/ses_forecast_november.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 그래프: 학습기간 실제값 + 11월 실제값 vs SES 예측값
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(train.index, train.values, color="#8A8F98", linewidth=1, label="학습기간 실제값(2011-01~10)")
ax.plot(test.index, test.values, color="#2E6F95", linewidth=1.6, label="11월 실제값")
ax.plot(forecast.index, forecast.values, color="#C1440E", linewidth=2, linestyle="--", label="SES 예측값")
ax.axvline(train.index.max(), color="#D9D9D9", linewidth=1)
ax.set_title(f"단순지수평활법(SES) 매출 예측 — alpha={alpha:.4f}, 초기수준={initial_level:,.0f}")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/ses_forecast_november.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 그래프: 11월만 확대 (실제값 vs 예측값 비교가 잘 보이도록)
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(test.index, test.values, color="#2E6F95", linewidth=1.8, marker="o", markersize=3, label="11월 실제값")
ax2.plot(forecast.index, forecast.values, color="#C1440E", linewidth=2, linestyle="--", label="SES 예측값(일정)")
ax2.set_title("11월 실제값 vs SES 예측값 (확대)")
ax2.set_xlabel("날짜")
ax2.set_ylabel("순매출(net_sales)")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False)
fig2.autofmt_xdate()
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/ses_forecast_november_zoom.png", dpi=150)
plt.close(fig2)

print(f"\n저장 완료: {TABLE_DIR}/ses_forecast_metrics.csv")
print(f"저장 완료: {TABLE_DIR}/ses_forecast_november.csv")
print(f"저장 완료: {FIG_DIR}/ses_forecast_november.png")
print(f"저장 완료: {FIG_DIR}/ses_forecast_november_zoom.png")
