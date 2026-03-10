import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

st.set_page_config(page_title="Dual-Constraint Quant", layout="wide")

# --- 세션 상태 ---
if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0

ETF_INFO = {
    "주식": {"SPY": "S&P500", "QQQ": "나스닥100", "SMH": "반도체", "XLK": "기술주", "LIT": "2차전지"},
    "채권": {"IEF": "중기채", "TLT": "장기채", "SHV": "초단기채"},
    "대안": {"GLD": "금", "USO": "원유", "VNQ": "리츠"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    data = yf.download(tickers + ['SPY', 'QQQ'], period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_dual_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret = target_ret_pct / 100.0
    max_mdd_allowed = target_mdd_pct / 100.0
    yrs = len(rets) / 252

    # 🌟 목적함수: 변동성(표준편차) 최소화
    def vol_fn(w): return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))

    # 🌟 제약조건 1: 연평균 수익률(CAGR) >= 목표
    def ret_cons(w):
        cum_ret = (1 + (rets @ w)).cumprod().iloc[-1]
        cagr = (cum_ret ** (1/yrs)) - 1
        return cagr - target_ret

    # 🌟 제약조건 2: MDD <= 사용자 한계치
    def mdd_cons(w):
        cum_rets = (1 + (rets @ w)).cumprod()
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        current_mdd = -drawdown.min()
        return max_mdd_allowed - current_mdd

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': ret_cons},
        {'type': 'ineq', 'fun': mdd_cons}
    ]
    
    res = minimize(vol_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.7)]*len(universe), constraints=cons)
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- 설문 화면 (수치 기반) ---
if st.session_state.page == 'survey':
    st.title("🎯 목표 및 위험 한도 설정")
    st.write("당신이 원하는 투자 결과와 감내 가능한 고통의 크기를 숫자로 알려주세요.")
    
    st.session_state.target_return = st.slider("1. 연간 목표 수익률 (%)", 5.0, 25.0, 12.0)
    st.session_state.target_mdd = st.slider("2. 감내 가능한 최대 하락률 (MDD, %)", 5.0, 40.0, 15.0)
    
    if st.button("AI 조합 최적화 시작 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'
        st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 이중 제약 최적화 결과")
    st.info(f"목표 수익률: **연 {st.session_state.target_return}%** | 감내 MDD: **-{st.session_state.target_mdd}%**")
    
    data = get_data(universe)
    wts = find_dual_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
    
    if wts:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("💡 최적 자산 배분")
            for t, w in wts.items(): st.write(f"**{t}**: {w}%")
        with col2:
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            st.plotly_chart(px.line(pd.DataFrame({"AI Portfolio": pv, "QQQ": norm['QQQ']}), title="지수 대비 성과 시뮬레이션"))
    else:
        st.error("⚠️ 해당 조건(고수익-저위험)은 현재 ETF 조합으로 달성이 불가능합니다. 수익률을 낮추거나 MDD 한도를 늘려주세요.")
        if st.button("설정 수정하러 가기"): 
            st.session_state.page = 'survey'; st.rerun()
