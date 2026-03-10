import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Fallback Quant Advisor", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 15.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 12.0

# --- [ETF 유니버스 정보] ---
ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500 지수", "QQQ": "나스닥100 지수", "DIA": "다우존스 지수", "IWM": "러셀2000 지수"},
    "성장 섹터": {"SMH": "반도체", "XLK": "기술주", "VGT": "IT전체", "IBB": "바이오", "LIT": "2차전지", "SKYY": "클라우드"},
    "가치/배당 섹터": {"SCHD": "배당성장", "VYM": "고배당", "VIG": "배당귀족", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어", "XLP": "필수소비재", "XLU": "유틸리티", "XLRE": "리츠"},
    "안전자산": {"SHV": "초단기 국채", "IEF": "중기 국채", "TLT": "장기 국채", "GLD": "금 현물", "SLV": "은 현물", "BND": "종합 채권"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

def get_etf_details(ticker):
    for sector, etfs in ETF_INFO.items():
        if ticker in etfs: return sector, etfs[ticker]
    return "기타", ""

@st.cache_data(ttl=3600)
def get_data(tickers):
    data = yf.download(list(set(tickers + ['SPY', 'QQQ'])), period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_robust_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret, target_mdd = target_ret_pct / 100.0, target_mdd_pct / 100.0
    yrs, tol = len(rets) / 252, 0.01

    # 1. MDD를 목적 함수로 설정 (최소화 대상)
    def mdd_fn(w):
        cum_rets = (1 + (rets @ w)).cumprod()
        return -((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()

    # 제약조건: 수익률은 타겟 근처(±1%)여야 함
    def ret_cons(w):
        cagr = ((1 + (rets @ w)).cumprod().iloc[-1] ** (1/yrs)) - 1
        return 0.01 - abs(cagr - target_ret)

    # 🌟 시도 1: 수익률 & MDD 조건을 모두 만족하는지 확인
    def mdd_limit_cons(w):
        return target_mdd - mdd_fn(w)

    cons_full = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                 {'type': 'ineq', 'fun': ret_cons},
                 {'type': 'ineq', 'fun': mdd_limit_cons}]
    
    res = minimize(mdd_fn, [1./len(universe)]*len(universe), bounds=[(0, 0.4)]*len(universe), constraints=cons_full, method='SLSQP')

    # 🌟 시도 2 (Fallback): MDD 조건을 못 맞추면, 수익률만 맞춘 채 MDD를 '최소화'함
    is_fallback = False
    if not res.success:
        is_fallback = True
        cons_fallback = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                         {'type': 'ineq', 'fun': ret_cons}]
        res = minimize(mdd_fn, [1./len(universe)]*len(universe), bounds=[(0, 0.4)]*len(universe), constraints=cons_fallback, method='SLSQP')

    if not res.success: return None, False
    
    weights = {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}
    return weights, is_fallback

# --- UI ---
if st.session_state.page == 'survey':
    st.title("🏛️ 스마트 차선책 퀀트 엔진")
    col1, col2 = st.columns(2)
    st.session_state.target_return = col1.number_input("목표 수익률 (%)", 1.0, 30.0, 15.0)
    st.session_state.target_mdd = col2.number_input("목표 MDD (%)", 1.0, 50.0, 12.0)
    if st.button("분석 시작 🚀", use_container_width=True): st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 포트폴리오 분석 결과")
    data = get_data(universe)
    wts, is_fallback = find_robust_optimal(st.session_state.target_return, st.session_state.target_mdd, data)

    if wts:
        if is_fallback:
            st.warning(f"⚠️ 알림: 입력하신 MDD(-{st.session_state.target_mdd}%) 내에서는 목표 수익률 달성이 불가능합니다. 수익률 {st.session_state.target_return}%를 유지하면서 '가장 안전한(최저 MDD)' 조합을 찾아내었습니다.")
        else:
            st.success(f"✅ 축하합니다! 목표 수익률과 MDD 조건을 모두 만족하는 최적의 조합을 찾았습니다.")

        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("💡 추천 비중")
            for t, w in sorted(wts.items(), key=lambda x: x[1], reverse=True):
                sec, desc = get_etf_details(t)
                st.markdown(f"""<div style="padding: 6px; border-bottom: 1px solid #eee;">
                    <b>{t}</b> <span style="color:green;">{w}%</span><br>
                    <small>{sec} · {desc}</small></div>""", unsafe_allow_html=True)
        with col2:
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            df_plot = pd.DataFrame({"Portfolio": pv, "SPY": norm['SPY'], "QQQ": norm['QQQ']})
            st.plotly_chart(px.line(df_plot, title="성과 시뮬레이션"), use_container_width=True)
            
            res_list = []
            for c in df_plot.columns:
                yrs = len(df_plot[c])/252
                cagr, mdd = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100, ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                res_list.append({"구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
            st.table(pd.DataFrame(res_list))
    else:
        st.error("데이터 부족으로 결과를 낼 수 없습니다.")
