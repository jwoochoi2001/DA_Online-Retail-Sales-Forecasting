"""국가(Country)별 매출 분석. data/processed/transactions_clean.csv(제외규칙 통과 후 거래단위)를 사용."""
import sys

import matplotlib.pyplot as plt
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "data/processed/transactions_clean.csv"
TABLE_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"

df = pd.read_csv(IN_PATH)
df["is_cancel"] = df["is_cancel"].astype(bool)
non_cancel = df.loc[~df["is_cancel"]]
cancel = df.loc[df["is_cancel"]]

summary = df.groupby("Country")["amount"].sum().rename("net_sales").to_frame()
summary["gross_sales"] = non_cancel.groupby("Country")["amount"].sum()
summary["cancel_amount"] = cancel.groupby("Country")["amount"].sum()
summary["n_invoices"] = non_cancel.groupby("Country")["InvoiceNo"].nunique()
summary["n_customers"] = non_cancel.groupby("Country")["CustomerID"].nunique()
summary["quantity"] = non_cancel.groupby("Country")["Quantity"].sum()
summary = summary.fillna(0)
summary["avg_order_value"] = (summary["gross_sales"] / summary["n_invoices"]).round(2)
summary["share_%"] = (summary["net_sales"] / summary["net_sales"].sum() * 100).round(2)

summary = summary.sort_values("net_sales", ascending=False)
summary[["net_sales", "gross_sales", "cancel_amount", "avg_order_value"]] = \
    summary[["net_sales", "gross_sales", "cancel_amount", "avg_order_value"]].round(2)

print(f"국가 수: {summary.shape[0]}")
print(summary.head(15).to_string())

summary.to_csv(f"{TABLE_DIR}/country_sales_summary.csv", encoding="utf-8-sig")
print(f"\n저장 완료: {TABLE_DIR}/country_sales_summary.csv")

uk_share = summary.loc["United Kingdom", "share_%"]
print(f"\nUnited Kingdom 매출 비중: {uk_share}% (나머지 {37}개국 합계 {100 - uk_share:.2f}%)")

# ------------------------------------------------------------------
# 그래프 1: 매출 상위 10개국 (UK 제외 나머지 비교가 잘 보이도록 별도 표시)
# ------------------------------------------------------------------
top10 = summary.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#C1440E" if c == "United Kingdom" else "#2E6F95" for c in top10.index]
bars = ax.barh(top10.index, top10["net_sales"], color=colors)
ax.set_title("국가별 순매출 상위 10개국")
ax.set_xlabel("순매출(net_sales)")
ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, top10["net_sales"]):
    ax.annotate(f"{val:,.0f}", (val, bar.get_y() + bar.get_height() / 2),
                textcoords="offset points", xytext=(5, 0), va="center", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/country_top10_net_sales.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# 그래프 2: UK 제외 상위 10개국 (UK 영향 제거하고 나머지 국가끼리 비교)
# ------------------------------------------------------------------
top10_ex_uk = summary.drop(index="United Kingdom").head(10).iloc[::-1]
fig2, ax2 = plt.subplots(figsize=(9, 6))
bars2 = ax2.barh(top10_ex_uk.index, top10_ex_uk["net_sales"], color="#2E6F95")
ax2.set_title("국가별 순매출 상위 10개국 (영국 제외)")
ax2.set_xlabel("순매출(net_sales)")
ax2.grid(axis="x", color="#D9D9D9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars2, top10_ex_uk["net_sales"]):
    ax2.annotate(f"{val:,.0f}", (val, bar.get_y() + bar.get_height() / 2),
                 textcoords="offset points", xytext=(5, 0), va="center", fontsize=8)
fig2.tight_layout()
fig2.savefig(f"{FIG_DIR}/country_top10_net_sales_ex_uk.png", dpi=150)
plt.close(fig2)

print(f"저장 완료: {FIG_DIR}/country_top10_net_sales.png")
print(f"저장 완료: {FIG_DIR}/country_top10_net_sales_ex_uk.png")
