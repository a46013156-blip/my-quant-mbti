import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Precision Quant Portfolio", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 15.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 12.0

# --- [ETF 유니버스 및 섹터 코멘트] ---
ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500 지수 추종", "QQQ": "나스닥100 지수 추종", "DIA": "다우존스 지수 추종", "IWM": "러셀2000 지수 추종"},
    "성장 섹터": {"SMH": "글로벌 반도체 기업", "XLK": "빅테크 및 IT 인프라", "VGT": "정보 기술 전반", "IBB": "바이오 테크 혁신", "LIT": "리튬 및 2차전지", "SKYY": "클라우드 컴퓨팅"},
    "가치/배당 섹터": {"SCHD": "배당성장 우량주", "VYM": "고배당 가치주", "VIG": "배당귀족주", "XLF": "금융/은행", "XLE": "에너지/원유", "XLV": "헬스케어/제약", "XLP": "필수소비재", "XLU": "유틸리티", "XLRE": "리츠/부동산"},
    "안전자산": {"SHV": "초단기 국채(현금성)", "IEF": "중기 국채", "TLT": "장기 국채", "GLD": "금 현물", "SLV": "은 현물", "BND": "종합 채권 시장"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

def get_etf_details(ticker):
    for sector, etfs in ETF_INFO.items():
        if ticker in etfs: return sector, etfs[ticker]
    return "기타", "상세 정보 없음"

@st.cache_data(ttl=3600)
def get_data(tickers):
    all_t = list(set(tickers + ['SPY', 'QQQ']))
    data = yf.download(all_t, period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_precise_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret = target_ret_pct / 100.0
    target_mdd = target_mdd_pct / 100.0
    yrs = len(rets) / 252
    tol = 0.01 # 🌟 오차범위 1% (상하 총 2%)

    # 목적함수: 변동성 최소화
    def vol_fn(w): return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))

    # 🌟 제약조건 1: 수익률이 (목표 - 1%) ~ (목표 + 1%) 사이에 있어야 함
    def ret_range_cons(w):
        cagr = ((1 + (rets @ w)).cumprod().iloc[-1] ** (1/yrs)) - 1
        return tol - abs(cagr - target_ret)

    # 🌟 제약조건 2: MDD가 (목표 - 1%) ~ (목표 + 1%) 사이에 있어야 함
    def mdd_range_cons(w):
        cum_rets = (1 + (rets @ w)).cumprod()
        mdd = -((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()
        return tol - abs(mdd - target_mdd)

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': ret_range_cons},
            {'type': 'ineq', 'fun': mdd_range_cons}]
    
    res = minimize(vol_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.4)]*len(universe), constraints=cons, method='SLSQP')
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- UI 레이아웃 ---
if st.session_state.page == 'survey':
    st.title("🏛️ 정밀 타겟 퀀트 엔진")
    st.write("수익률과 MDD를 설정하신 수치의 **±1%(총 2% 오차범위)** 내로 조절하여 최적화합니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 30.0, float(st.session_state.target_return), 0.1)
    with col2:
        st.session_state.target_mdd = st.number_input("목표 MDD (%)", 1.0, 50.0, float(st.session_state.target_mdd), 0.1)
    
    if st.button("오차범위 정밀 분석 및 포트폴리오 생성 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 퀀트 포트폴리오 성과 리포트")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32;">
        <b>🎯 타겟 조건 (오차범위 2%)</b> | 목표 수익률: <b>{st.session_state.target_return}%</b> | 목표 MDD: <b>-{st.session_state.target_mdd}%</b>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if st.button("⬅️ 설정 수정"): st.session_state.page = 'survey'; st.rerun()

    data = get_data(universe)
    wts = find_precise_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
    
    if wts:
        col1, col2 = st.columns([1.3, 2.5])
        with col1:
            st.subheader("💡 추천 종목 및 섹터 코멘트")
            sorted_wts = sorted(wts.items(), key=lambda x: x[1], reverse=True)
            for t, w in sorted_wts:
                sector, comment = get_etf_details(t)
                st.markdown(f"**{t}** ({w}%)")
                st.caption(f"**섹터:** {sector} | **설명:** {comment}")
                st.write("---")
        with col2:
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            df_plot = pd.DataFrame({"추천 포트폴리오": pv, "S&P 500 (SPY)": norm['SPY'], "나스닥 100 (QQQ)": norm['QQQ']})
            st.plotly_chart(px.line(df_plot, title="과거 10년 성과 비교 시뮬레이션"), use_container_width=True)
            
            # 성과 지표 비교표
            res_list = []
            for c in df_plot.columns:
                yrs = len(df_plot[c])/252
                cagr = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100
                mdd = ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                res_list.append({"자산 구분": c, "CAGR(수익률)": f"{cagr:.2f}%", "MDD(최대낙폭)": f"{mdd:.2f}%"})
            st.subheader("📊 지수 대비 성과 요약")
            st.table(pd.DataFrame(res_list))
    else:
        st.error("⚠️ 해당 조건(2% 오차범위 내)을 만족하는 조합을 찾을 수 없습니다. 목표를 현실적으로 조정해주세요.")
