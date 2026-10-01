"""일별 매출 시계열 전처리 (report/전처리_기준.md 기준).

data/raw/Online Retail.xlsx -> data/processed/daily_sales.csv
                              -> outputs/tables/preprocessing_exclusion_log.csv
원본은 읽기만 하고 수정하지 않는다. 같은 입력으로 다시 실행하면 같은 결과가 나온다.
"""
import sys

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH = "data/raw/Online Retail.xlsx"
PROCESSED_PATH = "data/processed/daily_sales.csv"
LOG_PATH = "outputs/tables/preprocessing_exclusion_log.csv"

PERIOD_START = "2010-12-01"
PERIOD_END = "2011-12-08"

UK_HOLIDAYS = pd.to_datetime([
    "2010-12-27", "2010-12-28", "2011-01-03",
    "2011-04-22", "2011-04-25", "2011-04-29",
    "2011-05-02", "2011-05-30", "2011-08-29",
])

# ------------------------------------------------------------------
# 1. 불러오기와 자료형
# ------------------------------------------------------------------
raw = pd.read_excel(RAW_PATH, sheet_name="Online Retail")

df = pd.DataFrame()
df["InvoiceNo"] = raw["InvoiceNo"].astype(str).str.strip()
df["StockCode"] = raw["StockCode"].astype(str).str.strip()
df["Description"] = raw["Description"].astype(str).str.strip()
df.loc[raw["Description"].isna(), "Description"] = pd.NA
df["Quantity"] = raw["Quantity"].astype("int64")
df["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
df["UnitPrice"] = raw["UnitPrice"].astype("float64")
df["CustomerID"] = raw["CustomerID"].apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)
df["Country"] = raw["Country"].astype(str).str.strip()

# 0. 용어와 기본 계산
df["amount"] = df["Quantity"] * df["UnitPrice"]
df["is_cancel"] = df["InvoiceNo"].str.startswith("C")
df["date"] = df["InvoiceDate"].dt.normalize()

# is_closed 판정용: 제외 규칙 적용 전 원본 날짜 집합
raw_dates_with_rows = set(df["date"].dt.normalize().unique())

print(f"원본 로드: {len(df):,}행, amount 합 {df['amount'].sum():,.2f}")

# ------------------------------------------------------------------
# 2. 행 제외 규칙 (순서대로 적용, 로그 기록)
# ------------------------------------------------------------------
code_upper = df["StockCode"].str.upper()

log_rows = []


def apply_step(remaining, mask_exclude, step, rule):
    """remaining(현재 남은 행)에서 mask_exclude(제외할 행)를 걷어내고 로그 남김."""
    excluded = remaining.loc[mask_exclude]
    kept = remaining.loc[~mask_exclude]
    log_rows.append({
        "단계": step, "규칙": rule,
        "제외 건수": len(excluded), "제외 금액": round(excluded["amount"].sum(), 2),
        "남은 행 수": len(kept), "남은 금액": round(kept["amount"].sum(), 2),
    })
    return kept


remaining = df
remaining = apply_step(remaining, remaining.duplicated(
    subset=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
    keep="first"), "2-1", "완전 중복 행")

acct_codes = {"AMAZONFEE", "BANK CHARGES", "CRUK", "B"}
remaining = apply_step(remaining, remaining["StockCode"].str.upper().isin(acct_codes), "2-2", "회계 조정 코드")

manual_codes = {"M", "D", "S"}
remaining = apply_step(remaining, remaining["StockCode"].str.upper().isin(manual_codes), "2-3", "수기 정정·할인·샘플")

shipping_codes = {"POST", "DOT", "C2"}
remaining = apply_step(remaining, remaining["StockCode"].str.upper().isin(shipping_codes), "2-4", "배송비")

remaining = apply_step(remaining, remaining["StockCode"].str.upper().str.startswith("GIFT_"), "2-5", "상품권 판매")

remaining = apply_step(remaining, remaining["UnitPrice"] <= 0, "2-6", "단가 0 이하")

remaining = apply_step(remaining, (remaining["Quantity"] < 0) & (~remaining["is_cancel"]), "2-7", "취소 아닌 음수 수량")

remaining = apply_step(remaining, remaining["InvoiceDate"] >= pd.Timestamp("2011-12-09 00:00"), "2-8", "불완전한 마지막 날")

log_df = pd.DataFrame(log_rows)
print(log_df.to_string(index=False))
print(f"\n남는 행: {len(remaining):,}행, amount 합 {remaining['amount'].sum():,.2f}")

# 제외 규칙(2단계)까지 통과한 거래 단위 데이터. 국가별 등 날짜 외 다른 축의 분석에 재사용.
remaining.to_csv("data/processed/transactions_clean.csv", index=False, encoding="utf-8-sig")
print("저장 완료: data/processed/transactions_clean.csv (거래 단위, 제외규칙 적용 후)")

# ------------------------------------------------------------------
# 3. 일별 집계
# ------------------------------------------------------------------
non_cancel = remaining.loc[~remaining["is_cancel"]]
cancel = remaining.loc[remaining["is_cancel"]]

daily = remaining.groupby("date")["amount"].sum().rename("net_sales").to_frame()
daily["gross_sales"] = non_cancel.groupby("date")["amount"].sum()
daily["cancel_amount"] = cancel.groupby("date")["amount"].sum()
daily["n_invoices"] = non_cancel.groupby("date")["InvoiceNo"].nunique()
daily["n_cancel_invoices"] = cancel.groupby("date")["InvoiceNo"].nunique()
daily["n_lines"] = non_cancel.groupby("date").size()
daily["quantity"] = non_cancel.groupby("date")["Quantity"].sum()
daily["n_customers"] = non_cancel.groupby("date")["CustomerID"].nunique()

daily = daily.fillna({"gross_sales": 0, "cancel_amount": 0, "n_invoices": 0, "n_cancel_invoices": 0,
                       "n_lines": 0, "quantity": 0, "n_customers": 0})

check_diff = abs(daily["net_sales"].sum() - remaining["amount"].sum())
print(f"\n검산(일별 net_sales 합 - 남은행 amount 합): {check_diff:.8f} (허용오차 1e-6)")
assert check_diff < 1e-6, "일별 집계 검산 실패"

# ------------------------------------------------------------------
# 4. 달력 채우기와 날짜 변수
# ------------------------------------------------------------------
full_range = pd.date_range(start=PERIOD_START, end=PERIOD_END, freq="D")
daily = daily.reindex(full_range)
amount_count_cols = ["net_sales", "gross_sales", "cancel_amount", "n_invoices", "n_cancel_invoices",
                      "n_lines", "quantity", "n_customers"]
daily[amount_count_cols] = daily[amount_count_cols].fillna(0)
int_cols = ["n_invoices", "n_cancel_invoices", "n_lines", "quantity", "n_customers"]
daily[int_cols] = daily[int_cols].astype("int64")
daily.index.name = "date"
daily = daily.reset_index()

daily["is_closed"] = (~daily["date"].isin(raw_dates_with_rows)).astype(int)

n_closed = int(daily["is_closed"].sum())
print(f"\n영업일 {len(daily) - n_closed}일, 휴무일 {n_closed}일 (전체 {len(daily)}일)")

daily["dow"] = daily["date"].dt.dayofweek
daily["is_sunday"] = (daily["dow"] == 6).astype(int)
daily["is_uk_holiday"] = daily["date"].isin(UK_HOLIDAYS).astype(int)
daily["year_month"] = daily["date"].dt.strftime("%Y-%m")

partial_lookup = {}
for ym in daily["year_month"].unique():
    n_in_period = (daily["year_month"] == ym).sum()
    n_in_calendar = pd.Period(ym, freq="M").days_in_month
    partial_lookup[ym] = int(n_in_period < n_in_calendar)
daily["is_partial_month"] = daily["year_month"].map(partial_lookup)

# ------------------------------------------------------------------
# 5. 이상치 표시 (지우지 않음, 영업일만 대상)
# ------------------------------------------------------------------
daily["is_outlier"] = 0
business = daily.loc[daily["is_closed"] == 0].copy()
business["y"] = np.log(business["net_sales"])

med = business.groupby("dow")["y"].transform("median")
mad = business.groupby("dow")["y"].transform(lambda s: (s - s.median()).abs().median())
z = 0.6745 * (business["y"] - med) / mad
business["is_outlier"] = (z.abs() > 3.5).astype(int)

daily.loc[business.index, "is_outlier"] = business["is_outlier"]

n_outlier = int(daily["is_outlier"].sum())
print(f"이상치(|z|>3.5) 영업일: {n_outlier}건")

# ------------------------------------------------------------------
# 저장
# ------------------------------------------------------------------
daily.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {PROCESSED_PATH} ({len(daily)}행)")
print(f"저장 완료: {LOG_PATH}")
