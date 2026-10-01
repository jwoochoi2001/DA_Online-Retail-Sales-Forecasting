"""일별 순매출(net_sales)에 7일 후행 이동평균과 14일 후행 이동평균을 계산해 그린다.
후행(trailing) 이동평균: 당일 + 이전 (n-1)일의 평균. 창 길이만큼 앞부분은 비교 대상이 없어 빈 값(NaN)."""
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

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["net_sales_ma7"] = df["net_sales"].rolling(window=7, min_periods=7).mean()
df["net_sales_ma14"] = df["net_sales"].rolling(window=14, min_periods=14).mean()

df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(df[["date", "net_sales", "net_sales_ma7", "net_sales_ma14"]].head(16).to_string(index=False))
print(f"\nnet_sales_ma7 빈 값: {df['net_sales_ma7'].isna().sum()}건, "
      f"net_sales_ma14 빈 값: {df['net_sales_ma14'].isna().sum()}건")

# ------------------------------------------------------------------
# 7일 이동평균 그래프
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(df["date"], df["net_sales"], color="#8A8F98", linewidth=1, label="일별 순매출(실제값)")
ax.plot(df["date"], df["net_sales_ma7"], color="#2E6F95", linewidth=2, label="7일 후행 이동평균")
ax.set_title("일별 순매출 실제값 vs 7일 후행 이동평균")
ax.set_xlabel("날짜")
ax.set_ylabel("순매출(net_sales)")
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/net_sales_ma7.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 7일 vs 14일 이동평균 비교 그래프 (창이 길수록 더 평평해지는 것을 보여줌)
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(14, 5.5))
ax2.plot(df["date"], df["net_sales"], color="#D9D9D9", linewidth=1, label="일별 순매출(실제값)")
ax2.plot(df["date"], df["net_sales_ma7"], color="#2E6F95", linewidth=1.8, label="7일 후행 이동평균")
ax2.plot(df["date"], df["net_sales_ma14"], color="#C1440E", linewidth=1.8, label="14일 후행 이동평균")
ax2.set_title("이동평균 창 길이 비교: 7일 vs 14일")
ax2.set_xlabel("날짜")
ax2.set_ylabel("순매출(net_sales)")
ax2.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(frameon=False, loc="upper left")
fig2.autofmt_xdate()
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/net_sales_ma7_vs_ma14.png", dpi=150)
plt.close(fig2)

print(f"\n저장 완료: {OUT_PATH} (net_sales_ma7, net_sales_ma14 칼럼 추가)")
print(f"저장 완료: {FIG_DIR}/net_sales_ma7.png")
print(f"저장 완료: {FIG_DIR}/net_sales_ma7_vs_ma14.png")
