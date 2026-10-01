"""SARIMA로 net_sales를 학습(2011-01-01~2011-10-31)하고 11월을 예측, SES/Holt/Holt-Winters와 비교.
d는 ADF 정상성 검정으로 결정(비정상 -> d=1), 계절 주기는 ACF에서 확인한 7일, D=1로 고정.
p,q,P,Q는 학습 자료에서 AIC 최소화로 자동 탐색."""
import itertools
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

IN_PATH = "data/processed/daily_sales.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"
M = 7  # 주간 계절 주기

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df.index.freq = "D"

train = df.loc["2011-01-01":"2011-10-31", "net_sales"]
test = df.loc["2011-11-01":"2011-11-30", "net_sales"]
print(f"학습: {train.index.min().date()} ~ {train.index.max().date()} ({len(train)}일)")
print(f"검증: {test.index.min().date()} ~ {test.index.max().date()} ({len(test)}일)")


def eval_forecast(actual, forecast):
    mae = (actual - forecast).abs().mean()
    rmse = np.sqrt(((actual - forecast) ** 2).mean())
    mask = actual != 0
    mape = ((actual[mask] - forecast[mask]).abs() / actual[mask]).mean() * 100
    return mae, rmse, mape


# ------------------------------------------------------------------
# d 결정: ADF 정상성 검정
# ------------------------------------------------------------------
adf_raw_p = adfuller(train, autolag="AIC")[1]
adf_diff_p = adfuller(train.diff().dropna(), autolag="AIC")[1]
d = 0 if adf_raw_p < 0.05 else 1
print(f"\nADF(원계열) p={adf_raw_p:.4f} -> {'정상' if d == 0 else '비정상'}, d={d} 사용")
print(f"ADF(1차차분) p={adf_diff_p:.4f}")

# ------------------------------------------------------------------
# p,q,P,Q 격자탐색 (AIC 최소화), D=1, m=7 고정
# ------------------------------------------------------------------
D = 1
best_aic = np.inf
best_order = None
search_rows = []

for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, M),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        search_rows.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "m": M, "AIC": fit.aic})
        if fit.aic < best_aic:
            best_aic = fit.aic
            best_order = (p, d, q, P, D, Q)
    except Exception:
        continue

search_df = pd.DataFrame(search_rows).sort_values("AIC")
search_df.to_csv(f"{TABLE_DIR}/sarima_order_search.csv", index=False, encoding="utf-8-sig")
print(f"\n격자탐색 {len(search_df)}개 조합 완료, 최적: order={best_order[:3]}, seasonal_order={best_order[3:]}"
      f"m={M}, AIC={best_aic:.2f}")
print(search_df.head(5).round(2).to_string(index=False))

p, d, q, P, D, Q = best_order
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, M),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_forecast = sarima_model.get_forecast(len(test))
sarima_mean = sarima_forecast.predicted_mean
sarima_mean.index = test.index
sarima_ci = sarima_forecast.conf_int(alpha=0.05)
sarima_ci.index = test.index

# ------------------------------------------------------------------
# 비교용 SES / Holt / Holt-Winters (기존과 동일 설정으로 재적합)
# ------------------------------------------------------------------
ses_model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
ses_forecast = ses_model.forecast(len(test))
ses_forecast.index = test.index

holt_model = Holt(train, damped_trend=False, initialization_method="estimated").fit(optimized=True)
holt_forecast = holt_model.forecast(len(test))
holt_forecast.index = test.index

hw_model = ExponentialSmoothing(train, trend="add", damped_trend=False, seasonal="add", seasonal_periods=7,
                                 initialization_method="estimated").fit(optimized=True)
hw_forecast = hw_model.forecast(len(test))
hw_forecast.index = test.index

# ------------------------------------------------------------------
# 정확도 비교
# ------------------------------------------------------------------
rows = []
for name, fc in [("SES", ses_forecast), ("Holt", holt_forecast),
                  ("Holt-Winters", hw_forecast), ("SARIMA", sarima_mean)]:
    mae, rmse, mape = eval_forecast(test, fc)
    rows.append({"model": name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{TABLE_DIR}/all_models_forecast_metrics.csv", index=False, encoding="utf-8-sig")

print(f"\nSARIMA order=({p},{d},{q})({P},{D},{Q}){M}")
print(metrics.round(3).to_string(index=False))

result = pd.DataFrame({
    "actual": test, "SES": ses_forecast, "Holt": holt_forecast,
    "Holt-Winters": hw_forecast, "SARIMA": sarima_mean,
})
result.to_csv(f"{TABLE_DIR}/all_models_forecast_november.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------
# 그래프 1: 전체 기간 + 11월 4개 모델 예측
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(train.index, train.values, color="#D9D9D9", linewidth=1, label="학습기간 실제값")
ax.plot(test.index, test.values, color="#2E6F95", linewidth=1.8, label="11월 실제값")
ax.plot(ses_forecast.index, ses_forecast.values, color="#B0B4BA", linewidth=1.4, linestyle="--", label="SES")
ax.plot(holt_forecast.index, holt_forecast.values, color="#8A8F98", linewidth=1.4, linestyle="--", label="Holt")
ax.plot(hw_forecast.index, hw_forecast.values, color="#5FA8D3", linewidth=1.6, linestyle="--", label="Holt-Winters")
ax.plot(sarima_mean.index, sarima_mean.values, color="#C1440E", linewidth=2.2, linestyle="--", label="SARIMA")
ax.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color="#C1440E", alpha=0.12,
                 label="SARIMA 95% 신뢰구간")
ax.axvline(train.index.max(), color="#D9D9D9", linewidth=1)
ax.set_title("SES vs Holt vs Holt-Winters vs SARIMA 매출 예측 비교")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/all_models_forecast_november.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 그래프 2: 11월 확대
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
ax2.plot(test.index, test.values, color="#2E6F95", linewidth=1.8, marker="o", markersize=3, label="11월 실제값")
ax2.plot(ses_forecast.index, ses_forecast.values, color="#B0B4BA", linewidth=1.4, linestyle="--", label="SES")
ax2.plot(holt_forecast.index, holt_forecast.values, color="#8A8F98", linewidth=1.4, linestyle="--", label="Holt")
ax2.plot(hw_forecast.index, hw_forecast.values, color="#5FA8D3", linewidth=1.6, linestyle="--", label="Holt-Winters")
ax2.plot(sarima_mean.index, sarima_mean.values, color="#C1440E", linewidth=2.2, linestyle="--", marker="D",
          markersize=4, label="SARIMA")
ax2.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color="#C1440E", alpha=0.12)
ax2.set_title("11월 실제값 vs 4개 모델 예측 (확대)")
ax2.set_xlabel("날짜")
ax2.set_ylabel("순매출(net_sales)")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False, ncol=2)
fig2.autofmt_xdate()
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/all_models_forecast_november_zoom.png", dpi=150)
plt.close(fig2)

print(f"\n저장 완료: {TABLE_DIR}/sarima_order_search.csv")
print(f"저장 완료: {TABLE_DIR}/all_models_forecast_metrics.csv")
print(f"저장 완료: {TABLE_DIR}/all_models_forecast_november.csv")
print(f"저장 완료: {FIG_DIR}/all_models_forecast_november.png")
print(f"저장 완료: {FIG_DIR}/all_models_forecast_november_zoom.png")
