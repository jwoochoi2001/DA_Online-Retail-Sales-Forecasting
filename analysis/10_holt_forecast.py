"""Holt 선형추세법(감쇠 없음)으로 net_sales를 학습(2011-01-01~2011-10-31)하고 11월을 예측.
SES 예측과 함께 비교한다. 평활계수(alpha, beta)와 초기수준·초기추세는 학습 자료로 자동 추정."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

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


def eval_forecast(actual, forecast):
    mae = (actual - forecast).abs().mean()
    rmse = np.sqrt(((actual - forecast) ** 2).mean())
    mask = actual != 0
    mape = ((actual[mask] - forecast[mask]).abs() / actual[mask]).mean() * 100
    return mae, rmse, mape


# SES (비교용, 이전 단계와 동일 설정으로 재적합)
ses_model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
ses_forecast = ses_model.forecast(len(test))
ses_forecast.index = test.index
ses_alpha = ses_model.params["smoothing_level"]

# Holt 선형추세법: 감쇠추세 사용 안 함(damped_trend=False), alpha/beta/초기값 자동 추정
holt_model = Holt(train, damped_trend=False, initialization_method="estimated").fit(optimized=True)
holt_forecast = holt_model.forecast(len(test))
holt_forecast.index = test.index

holt_alpha = holt_model.params["smoothing_level"]
holt_beta = holt_model.params["smoothing_trend"]
holt_init_level = holt_model.params["initial_level"]
holt_init_trend = holt_model.params["initial_trend"]

print(f"학습: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")
print(f"검증: {test.index.min().date()} ~ {test.index.max().date()} ({len(test)}일)")

print(f"\n[Holt 선형추세법] alpha={holt_alpha:.4f}, beta={holt_beta:.4f}, "
      f"초기수준={holt_init_level:,.2f}, 초기추세={holt_init_trend:,.2f}")
print(f"Holt 11월 예측(첫날 {holt_forecast.iloc[0]:,.0f} -> 마지막날 {holt_forecast.iloc[-1]:,.0f})")

ses_mae, ses_rmse, ses_mape = eval_forecast(test, ses_forecast)
holt_mae, holt_rmse, holt_mape = eval_forecast(test, holt_forecast)

print(f"\n[SES]  MAE={ses_mae:,.2f}  RMSE={ses_rmse:,.2f}  MAPE={ses_mape:.2f}%")
print(f"[Holt] MAE={holt_mae:,.2f}  RMSE={holt_rmse:,.2f}  MAPE={holt_mape:.2f}%")

metrics = pd.DataFrame({
    "model": ["SES", "Holt(선형추세, 감쇠없음)"],
    "alpha": [ses_alpha, holt_alpha],
    "beta": [np.nan, holt_beta],
    "initial_level": [ses_model.params["initial_level"], holt_init_level],
    "initial_trend": [np.nan, holt_init_trend],
    "MAE": [ses_mae, holt_mae],
    "RMSE": [ses_rmse, holt_rmse],
    "MAPE_%": [ses_mape, holt_mape],
})
metrics.to_csv(f"{TABLE_DIR}/ses_holt_forecast_metrics.csv", index=False, encoding="utf-8-sig")
print(metrics.round(4).to_string(index=False))

result = pd.DataFrame({"actual": test, "ses_forecast": ses_forecast, "holt_forecast": holt_forecast})
result.to_csv(f"{TABLE_DIR}/ses_holt_forecast_november.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 그래프 1: 전체 기간 + 11월 SES vs Holt 예측
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(train.index, train.values, color="#D9D9D9", linewidth=1, label="학습기간 실제값(2011-01~10)")
ax.plot(test.index, test.values, color="#2E6F95", linewidth=1.6, label="11월 실제값")
ax.plot(ses_forecast.index, ses_forecast.values, color="#8A8F98", linewidth=2, linestyle="--", label="SES 예측값")
ax.plot(holt_forecast.index, holt_forecast.values, color="#C1440E", linewidth=2, linestyle="--", label="Holt 예측값")
ax.axvline(train.index.max(), color="#D9D9D9", linewidth=1)
ax.set_title("SES vs Holt(선형추세) 매출 예측 비교")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/ses_holt_forecast_november.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 그래프 2: 11월만 확대
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(test.index, test.values, color="#2E6F95", linewidth=1.8, marker="o", markersize=3, label="11월 실제값")
ax2.plot(ses_forecast.index, ses_forecast.values, color="#8A8F98", linewidth=2, linestyle="--", label="SES 예측값(수평)")
ax2.plot(holt_forecast.index, holt_forecast.values, color="#C1440E", linewidth=2, linestyle="--", label="Holt 예측값(직선 추세)")
ax2.set_title("11월 실제값 vs SES vs Holt 예측 (확대)")
ax2.set_xlabel("날짜")
ax2.set_ylabel("순매출(net_sales)")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False)
fig2.autofmt_xdate()
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/ses_holt_forecast_november_zoom.png", dpi=150)
plt.close(fig2)

print(f"\n저장 완료: {TABLE_DIR}/ses_holt_forecast_metrics.csv")
print(f"저장 완료: {TABLE_DIR}/ses_holt_forecast_november.csv")
print(f"저장 완료: {FIG_DIR}/ses_holt_forecast_november.png")
print(f"저장 완료: {FIG_DIR}/ses_holt_forecast_november_zoom.png")
