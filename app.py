"""온라인 소매 매출 분석 보고서 (Streamlit).
data/processed/*.csv, outputs/tables/*.csv는 읽기만 하고 수정하지 않는다.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

APP_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_MAIN = "#2E6F95"
COLOR_SUB = "#C1440E"
COLOR_GRAY = "#8A8F98"
FONT = "Malgun Gothic, sans-serif"

DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

st.set_page_config(page_title="온라인 소매 매출 분석 보고서", layout="wide")


def p(*parts):
    return os.path.join(APP_DIR, *parts)


@st.cache_data
def load_daily():
    df = pd.read_csv(p("data", "processed", "daily_sales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_table(name):
    return pd.read_csv(p("outputs", "tables", name))


@st.cache_data
def run_forecasts():
    """Jan-Oct를 학습해 남은 전체 구간(11월~12월 8일, 38일)을 예측한다. SES/Holt/Holt-Winters 비교."""
    daily_ = load_daily().set_index("date")
    daily_.index.freq = "D"
    train_ = daily_.loc["2011-01-01":"2011-10-31", "net_sales"]
    future_ = daily_.loc[daily_.index > train_.index.max(), "net_sales"]
    horizon = len(future_)

    ses_m = SimpleExpSmoothing(train_, initialization_method="estimated").fit(optimized=True)
    ses_fc = ses_m.forecast(horizon)
    ses_fc.index = future_.index

    holt_m = Holt(train_, damped_trend=False, initialization_method="estimated").fit(optimized=True)
    holt_fc = holt_m.forecast(horizon)
    holt_fc.index = future_.index

    hw_m = ExponentialSmoothing(
        train_, trend="add", damped_trend=False, seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_fc = hw_m.forecast(horizon)
    hw_fc.index = future_.index

    # SARIMA(1,1,2)(0,1,1)7: analysis/13_sarima_forecast.py의 AIC 격자탐색(36개 조합)에서 선정된 최적 차수
    sarima_m = SARIMAX(train_, order=(1, 1, 2), seasonal_order=(0, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_fc = sarima_m.forecast(horizon)
    sarima_fc.index = future_.index

    result_ = pd.DataFrame({
        "actual": future_, "SES": ses_fc, "Holt": holt_fc, "Holt-Winters": hw_fc, "SARIMA": sarima_fc,
    }).reset_index().rename(columns={"index": "date"})

    def metrics_for(mask, label):
        rows = []
        for col in ["SES", "Holt", "Holt-Winters", "SARIMA"]:
            actual = result_.loc[mask, "actual"]
            forecast = result_.loc[mask, col]
            mae = (actual - forecast).abs().mean()
            rmse = np.sqrt(((actual - forecast) ** 2).mean())
            nz = actual != 0
            mape = ((actual[nz] - forecast[nz]).abs() / actual[nz]).mean() * 100
            rows.append({"기간": label, "모델": col, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})
        return rows

    nov_mask = result_["date"].dt.month == 11
    dec_mask = result_["date"].dt.month == 12
    metrics_rows = metrics_for(nov_mask, "11월") + metrics_for(dec_mask, "12월(1~8일)")
    metrics_ = pd.DataFrame(metrics_rows)

    params_ = pd.DataFrame({
        "모델": ["SES", "Holt", "Holt-Winters", "SARIMA(1,1,2)(0,1,1)7"],
        "alpha": [ses_m.params["smoothing_level"], holt_m.params["smoothing_level"],
                  hw_m.params["smoothing_level"], np.nan],
        "beta": [np.nan, holt_m.params["smoothing_trend"], hw_m.params["smoothing_trend"], np.nan],
        "gamma": [np.nan, np.nan, hw_m.params["smoothing_seasonal"], np.nan],
        "AIC": [np.nan, np.nan, np.nan, sarima_m.aic],
    })

    return result_, metrics_.round(2), params_.round(4), train_.index.max()


def layout_ko(fig, **kwargs):
    fig.update_layout(font_family=FONT, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=40, r=20, b=40, l=50), **kwargs)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E5E5")
    return fig


daily = load_daily()
exclusion_log = load_table("preprocessing_exclusion_log.csv")
monthly_summary = load_table("monthly_net_sales_summary.csv")
weekday_summary = load_table("weekday_net_sales_summary.csv")
country_summary = load_table("country_sales_summary.csv")

st.title("온라인 소매 매출 분석 보고서")
st.caption(f"대상 기간: {daily['date'].min().date()} ~ {daily['date'].max().date()} "
           f"({len(daily)}일, 영업일 {int((daily['is_closed'] == 0).sum())}일 / 휴무일 {int((daily['is_closed'] == 1).sum())}일)")

with st.sidebar:
    st.header("필터")
    min_date, max_date = daily["date"].min().date(), daily["date"].max().date()
    date_range = st.date_input("기간", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    only_open = st.checkbox("휴무일 제외하고 보기", value=False)

mask = (daily["date"].dt.date >= start_date) & (daily["date"].dt.date <= end_date)
view = daily.loc[mask].copy()
if only_open:
    view = view.loc[view["is_closed"] == 0]

if len(view) == 0:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ------------------------------------------------------------------
# KPI
# ------------------------------------------------------------------
total = view["net_sales"].sum()
open_avg = view.loc[view["is_closed"] == 0, "net_sales"].mean()
max_row = view.loc[view["net_sales"].idxmax()]
n_outlier = int(view["is_outlier"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("기간 순매출 합계", f"{total:,.0f}")
c2.metric("영업일 평균", f"{open_avg:,.1f}")
c3.metric("최고 매출일", f"{max_row['date'].date()}", f"{max_row['net_sales']:,.0f}")
c4.metric("이상치 건수", f"{n_outlier}건")

(tab_overview, tab_trend, tab_month, tab_weekday, tab_country, tab_decomp, tab_forecast,
 tab_outlier) = st.tabs(
    ["전처리 요약", "일별 추이", "월별 분석", "요일 분석", "국가 분석", "시계열 분해", "예측 모형 비교", "이상치"]
)

# ------------------------------------------------------------------
# 전처리 요약
# ------------------------------------------------------------------
with tab_overview:
    st.subheader("행 제외 규칙 적용 로그 (Online Retail.xlsx 원본 → 정제)")
    st.dataframe(exclusion_log, use_container_width=True, hide_index=True)
    st.caption("제외 규칙은 원본 순서대로 누적 적용됨. report/전처리_기준.md 기준을 그대로 따름.")

    st.subheader("일별 집계 규칙")
    st.markdown(
        "- `net_sales` = 제외규칙 통과 후 남은 행(정상+취소) amount 합 — 분석·예측 대상\n"
        "- 휴무일(`is_closed`)은 원본 기준 해당 날짜 거래가 없으면 1, 값은 0으로 채움\n"
        "- 이상치(`is_outlier`)는 영업일만 대상으로 같은 요일끼리 log(net_sales)의 MAD 기준 `|z|>3.5`일 때 표시(삭제 아님)"
    )

# ------------------------------------------------------------------
# 일별 추이
# ------------------------------------------------------------------
with tab_trend:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["date"], y=view["net_sales"], mode="lines", name="실제 순매출",
                              line=dict(color=COLOR_GRAY, width=1),
                              hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=view["date"], y=view["net_sales_ma7"], mode="lines", name="7일 후행 이동평균",
                              line=dict(color=COLOR_MAIN, width=2)))
    fig.add_trace(go.Scatter(x=view["date"], y=view["net_sales_ma14"], mode="lines", name="14일 후행 이동평균",
                              line=dict(color=COLOR_SUB, width=2)))
    outlier_pts = view.loc[view["is_outlier"] == 1]
    fig.add_trace(go.Scatter(x=outlier_pts["date"], y=outlier_pts["net_sales"], mode="markers", name="이상치",
                              marker=dict(color=COLOR_SUB, size=10, symbol="diamond")))
    layout_ko(fig, title="일별 순매출 추이 및 이동평균", yaxis_title="순매출",
              hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 월별 분석
# ------------------------------------------------------------------
with tab_month:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_sum = go.Figure(go.Bar(x=monthly_summary["year_month"], y=monthly_summary["net_sales_sum"],
                                    marker_color=COLOR_MAIN,
                                    hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
        layout_ko(fig_sum, title="월별 순매출 합계", xaxis_title="월", yaxis_title="합계")
        st.plotly_chart(fig_sum, use_container_width=True)
    with col_b:
        fig_avg = go.Figure(go.Bar(x=monthly_summary["year_month"], y=monthly_summary["net_sales_daily_avg"],
                                    marker_color=COLOR_SUB,
                                    hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
        layout_ko(fig_avg, title="월별 하루평균 (합계 / 그달 날짜수)", xaxis_title="월", yaxis_title="하루평균")
        st.plotly_chart(fig_avg, use_container_width=True)

    st.dataframe(monthly_summary, use_container_width=True, hide_index=True)

    st.subheader("월별 실제값 vs 7일 이동평균")
    month_pick = st.selectbox("월 선택", sorted(daily["year_month"].unique()),
                               index=len(daily["year_month"].unique()) - 1)
    m_sub = daily.loc[daily["year_month"] == month_pick]
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=m_sub["day"], y=m_sub["net_sales"], mode="lines", name="실제값",
                                line=dict(color=COLOR_GRAY, width=1.2)))
    fig_m.add_trace(go.Scatter(x=m_sub["day"], y=m_sub["net_sales_ma7"], mode="lines", name="7일 이동평균",
                                line=dict(color=COLOR_MAIN, width=2)))
    layout_ko(fig_m, title=f"{month_pick} 일별 순매출", xaxis_title="일(day)", yaxis_title="순매출",
              legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_m, use_container_width=True)

# ------------------------------------------------------------------
# 요일 분석
# ------------------------------------------------------------------
with tab_weekday:
    ws = weekday_summary.set_index(weekday_summary.columns[0]).reindex(DOW_ORDER)

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=ws.index, y=ws["avg_all"], name="전체평균(휴무일 0 포함)", marker_color=COLOR_GRAY))
    fig_w.add_trace(go.Bar(x=ws.index, y=ws["avg_open"], name="영업일평균(휴무일 제외)", marker_color=COLOR_MAIN))
    fig_w.update_layout(barmode="group")
    layout_ko(fig_w, title="요일별 평균 순매출", xaxis_title="요일", yaxis_title="평균 순매출",
              legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("요일별 순매출 분포 (영업일만)")
    open_df = daily.loc[daily["is_closed"] == 0]
    fig_box = go.Figure()
    for d in DOW_ORDER:
        vals = open_df.loc[open_df["dow_label"] == d, "net_sales"]
        if len(vals) > 0:
            fig_box.add_trace(go.Box(y=vals, name=d, marker_color=COLOR_MAIN, boxmean=True))
    layout_ko(fig_box, title="요일별 순매출 상자그림 (영업일만)", yaxis_title="순매출", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.dataframe(ws, use_container_width=True)

# ------------------------------------------------------------------
# 국가 분석
# ------------------------------------------------------------------
with tab_country:
    top_n = st.slider("표시할 국가 수", 5, 20, 10)
    top = country_summary.sort_values("net_sales", ascending=False).head(top_n).iloc[::-1]
    colors = [COLOR_SUB if c == "United Kingdom" else COLOR_MAIN for c in top["Country"]]
    fig_c = go.Figure(go.Bar(x=top["net_sales"], y=top["Country"], orientation="h", marker_color=colors,
                              hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    layout_ko(fig_c, title=f"국가별 순매출 상위 {top_n}개국", xaxis_title="순매출")
    st.plotly_chart(fig_c, use_container_width=True)

    uk_share = country_summary.loc[country_summary["Country"] == "United Kingdom", "share_%"].iloc[0]
    st.caption(f"United Kingdom 비중 {uk_share:.2f}% — 나머지 {len(country_summary) - 1}개국 합계 {100 - uk_share:.2f}%")

    st.dataframe(country_summary.sort_values("net_sales", ascending=False), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# 시계열 분해
# ------------------------------------------------------------------
with tab_decomp:
    st.markdown("**분해식**: `net_sales = trend(7일 중심이동평균) + weekday_effect(요일효과, 합 0) + resid(잔차)`")

    month_pick2 = st.selectbox("월 선택 ", sorted(daily["year_month"].unique()),
                                index=len(daily["year_month"].unique()) - 1, key="decomp_month")
    d_sub = daily.loc[daily["year_month"] == month_pick2]

    fig_d = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("실제값", "trend (7일 중심이동평균)", "weekday_effect (요일효과)", "resid (잔차)"))
    fig_d.add_trace(go.Scatter(x=d_sub["day"], y=d_sub["net_sales"], line=dict(color=COLOR_GRAY, width=1.2)),
                     row=1, col=1)
    fig_d.add_trace(go.Scatter(x=d_sub["day"], y=d_sub["trend"], line=dict(color=COLOR_MAIN, width=1.5)),
                     row=2, col=1)
    fig_d.add_trace(go.Scatter(x=d_sub["day"], y=d_sub["weekday_effect"], line=dict(color=COLOR_SUB, width=1.2)),
                     row=3, col=1)
    fig_d.add_trace(go.Scatter(x=d_sub["day"], y=d_sub["resid"], line=dict(color=COLOR_GRAY, width=1.2)),
                     row=4, col=1)
    for r in (3, 4):
        fig_d.add_hline(y=0, line_color="#D9D9D9", line_width=1, row=r, col=1)
    fig_d.update_layout(height=750, showlegend=False)
    layout_ko(fig_d)
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("요일효과 (전체 기간, 합 0)")
    we = daily.drop_duplicates("dow_label")[["dow_label", "weekday_effect"]].set_index("dow_label").reindex(DOW_ORDER)
    st.dataframe(we.round(1), use_container_width=True)

# ------------------------------------------------------------------
# 예측 모형 비교 (SES / Holt / Holt-Winters)
# ------------------------------------------------------------------
with tab_forecast:
    fc_result, fc_metrics, fc_params, train_end = run_forecasts()

    st.markdown(
        f"**학습 구간**: 2011-01-01 ~ {train_end.date()} (304일) · "
        f"**예측 구간**: {fc_result['date'].min().date()} ~ {fc_result['date'].max().date()} "
        f"(11월 + 12월 1~8일, {len(fc_result)}일). SES/Holt/Holt-Winters는 학습 자료로 평활계수·초기값을, "
        f"SARIMA는 차수(1,1,2)(0,1,1)7을 analysis/13_sarima_forecast.py의 AIC 격자탐색으로 자동 결정."
    )

    st.subheader("모델별 평활계수")
    st.dataframe(fc_params, use_container_width=True, hide_index=True)

    st.subheader("예측 정확도 비교 (11월 vs 12월 1~8일)")
    st.dataframe(fc_metrics, use_container_width=True, hide_index=True)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=fc_result["date"], y=fc_result["actual"], mode="lines+markers",
                                 name="실제값", line=dict(color=COLOR_MAIN, width=1.8), marker=dict(size=4)))
    fig_fc.add_trace(go.Scatter(x=fc_result["date"], y=fc_result["SES"], mode="lines", name="SES",
                                 line=dict(color="#B0B4BA", width=1.6, dash="dash")))
    fig_fc.add_trace(go.Scatter(x=fc_result["date"], y=fc_result["Holt"], mode="lines", name="Holt",
                                 line=dict(color=COLOR_GRAY, width=1.6, dash="dash")))
    fig_fc.add_trace(go.Scatter(x=fc_result["date"], y=fc_result["Holt-Winters"], mode="lines", name="Holt-Winters",
                                 line=dict(color="#5FA8D3", width=1.8, dash="dash")))
    fig_fc.add_trace(go.Scatter(x=fc_result["date"], y=fc_result["SARIMA"], mode="lines", name="SARIMA",
                                 line=dict(color=COLOR_SUB, width=2.4, dash="dash")))
    fig_fc.add_vline(x=pd.Timestamp("2011-12-01"), line_color="#D9D9D9", line_width=1)
    layout_ko(fig_fc, title="11월~12월(1~8일) 실제값 vs 모델별 예측값", yaxis_title="순매출",
              hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.subheader("12월(1~8일) 예측 확대")
    dec_view = fc_result.loc[fc_result["date"].dt.month == 12]
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(x=dec_view["date"], y=dec_view["actual"], mode="lines+markers", name="실제값",
                                  line=dict(color=COLOR_MAIN, width=2), marker=dict(size=6)))
    fig_dec.add_trace(go.Scatter(x=dec_view["date"], y=dec_view["SES"], mode="lines+markers", name="SES",
                                  line=dict(color="#B0B4BA", width=1.6, dash="dash")))
    fig_dec.add_trace(go.Scatter(x=dec_view["date"], y=dec_view["Holt"], mode="lines+markers", name="Holt",
                                  line=dict(color=COLOR_GRAY, width=1.6, dash="dash")))
    fig_dec.add_trace(go.Scatter(x=dec_view["date"], y=dec_view["Holt-Winters"], mode="lines+markers",
                                  name="Holt-Winters", line=dict(color="#5FA8D3", width=1.8, dash="dash")))
    fig_dec.add_trace(go.Scatter(x=dec_view["date"], y=dec_view["SARIMA"], mode="lines+markers",
                                  name="SARIMA", line=dict(color=COLOR_SUB, width=2.4, dash="dash")))
    layout_ko(fig_dec, title="2011-12-01 ~ 12-08 실제값 vs 모델별 예측값", yaxis_title="순매출",
              hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_dec, use_container_width=True)

# ------------------------------------------------------------------
# 이상치
# ------------------------------------------------------------------
with tab_outlier:
    n = st.slider("표시할 이상치 개수", 1, 20, 5)
    top_out = daily.loc[daily["is_outlier"] == 1].copy()
    top_out["date"] = top_out["date"].dt.date
    show_cols = ["date", "net_sales", "trend", "weekday_effect", "resid", "dow_label", "is_uk_holiday"]
    show = top_out[show_cols].rename(columns={
        "date": "날짜", "net_sales": "순매출", "trend": "추세", "weekday_effect": "요일효과",
        "resid": "잔차", "dow_label": "요일", "is_uk_holiday": "공휴일",
    }).sort_values("잔차", key=abs, ascending=False).head(n).round(1)
    st.dataframe(show, use_container_width=True, hide_index=True)
