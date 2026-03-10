import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Custom Dual-Constraint Quant", layout="wide")

# --- 세션 상태 초기화 ---
if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0

# ETF 유니버스
ETF_INFO = {
    "주식": {"SPY": "S&P500", "QQQ": "나스닥100", "SMH": "반도체", "XLK": "기술주", "LIT": "2차전지"},
    "채권": {"IEF": "중기채", "TLT": "장기채", "SHV": "초단기채"},
    "대안": {"GLD": "금", "USO": "원유", "VNQ": "리츠"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    all_tickers = list(set(tickers + ['SPY', 'QQQ']))
    data = yf.download(all_tickers, period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def calculate_metrics(prices):
    """CAGR(연평균 수익률)과 MDD(최대 낙폭) 계산"""
    yrs = len(prices) / 252
    cagr = ((prices.iloc[-1] / prices.iloc[0]) ** (1 / yrs) - 1) * 100
    cum_rets = (prices / prices.iloc[0])
    drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
    mdd = drawdown.min() * 100
    return cagr, mdd

def find_dual_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret = target_ret_pct / 100.0
    max_mdd_allowed = target_mdd_pct / 100.0
    yrs = len(rets) / 252

    # 목적함수: 변동성 최소화
    def vol_fn(w): return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))

    # 제약조건 1: 연평균 수익률(CAGR) >= 목표
    def ret_cons(w):
        cum_ret = (1 + (rets @ w)).cumprod().iloc[-1]
        cagr = (cum_ret ** (1/yrs)) - 1
        return cagr - target_ret

    # 제약조건 2: MDD <= 사용자 한계치
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

# --- 입력 화면 (수치 직접 입력 방식) ---
if st.session_state.page == 'survey':
    st.title("🎯 목표 및 위험 한도 직접 설정")
    st.write("원하는 투자 조건을 수치로 정확하게 입력해 주세요.")
    
    col1, col2 = st.columns(2)
    with col1:
        # st.number_input을 사용하여 수치 직접 입력 가능하게 변경
        st.session_state.target_return = st.number_input(
            "1. 연간 목표 수익률 (%, 예: 15.0)", 
            min_value=1.0, max_value=50.0, 
            value=float(st.session_state.target_return), 
            step=0.5
        )
    with col2:
        st.session_state.target_mdd = st.number_input(
            "2. 감내 가능한 최대 하락률 (MDD, %, 예: 12.0)", 
            min_value=1.0, max_value=70.0, 
            value=float(st.session_state.target_mdd), 
            step=0.5
        )
    
    if st.button("AI 포트폴리오 최적 조합 찾기 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'
        st.rerun()

# --- 결과 대시보드 ---
elif st.session_state.page == 'dashboard':
    st.title("🛡️ 퀀트 포트폴리오 성과 리포트")
    st.info(f"설정 조건: 목표 수익률 **연 {st.session_state.target_return}%** | 허용 MDD **-{st.session_state.target_mdd}%**")
    
    if st.button("⬅️ 설정 다시 하기"):
        st.session_state.page = 'survey'
        st.rerun()

    data = get_data(universe)
    wts = find_dual_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
    
    if wts:
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("💡 최적 자산 배분 비중")
            # 틱커별 설명 찾기 함수
            def get_desc(t):
                for cat, items in ETF_INFO.items():
                    if t in items: return items[t]
                return ""

            for t, w in wts.items():
                st.write(f"**{t}** ({get_desc(t)}): {w}%")
        
        with col2:
            # 시뮬레이션 데이터 생성
            norm = (data / data.iloc[0]) * 100
            pv_prices = sum([norm[t] * (w/100) for t, w in wts.items()])
            
            df_plot = pd.DataFrame({
                "추천 포트폴리오": pv_prices,
                "S&P 500 (SPY)": norm['SPY'],
                "나스닥 100 (QQQ)": norm['QQQ']
            })
            
            st.plotly_chart(px.line(df_plot, title="과거 10년 성과 비교 시뮬레이션"), use_container_width=True)
            
            # --- 성과 지표 요약 표 (CAGR, MDD 비교) ---
            metrics_results = []
            for col in df_plot.columns:
                cagr, mdd = calculate_metrics(df_plot[col])
                metrics_results.append({
                    "자산 구분": col,
                    "연평균 수익률 (CAGR)": f"{cagr:.2f}%",
                    "최대 낙폭 (MDD)": f"{mdd:.2f}%"
                })
            
            st.subheader("📊 주요 성과 지표 비교표")
            st.table(pd.DataFrame(metrics_results))
    else:
        st.error("⚠️ 입력하신 조건(수익률/MDD)을 동시에 만족하는 조합이 현재 데이터상에 존재하지 않습니다. 목표 수익률을 낮추거나 허용 MDD를 늘려주세요.")
