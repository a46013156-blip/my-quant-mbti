import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# --- ETF 공통 데이터 및 함수 ---
TOLERANCE = 1.0 
ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500", "QQQ": "나스닥100", "DIA": "다우존스", "IWM": "러셀2000"},
    "성장 섹터": {"SMH": "반도체", "XLK": "기술주", "VGT": "IT전체", "XLV": "헬스케어", "XLF": "금융", "XLE": "에너지"},
    "배당/가치": {"SCHD": "배당성장", "VYM": "고배당", "VIG": "배당귀족", "DVY": "우량배당"},
    "안전자산/원자재": {"SHV": "초단기국채", "IEF": "중기국채", "TLT": "장기국채", "LQD": "회사채", "GLD": "금 현물", "SLV": "은 현물"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

def get_etf_details(ticker):
    for sector, etfs in ETF_INFO.items():
        if ticker in etfs: return sector, etfs[ticker]
    return "기타", ""

@st.cache_data(ttl=3600)
def get_data(tickers):
    try:
        data = yf.download(list(set(tickers + ['SPY', 'QQQ'])), period="10y", progress=False)['Close']
        valid_cols = data.columns[data.notna().sum() > (len(data) * 0.9)]
        return data[valid_cols].ffill().dropna()
    except: return pd.DataFrame()

def find_robust_optimal(target_ret_pct, target_mdd_pct, max_assets, max_gold_pct, data):
    if data.empty or len(data) < 20: return None, False
    current_universe = [t for t in universe if t in data.columns]
    rets = data[current_universe].pct_change().dropna()
    target_ret, target_mdd = target_ret_pct / 100.0, target_mdd_pct / 100.0
    yrs = len(rets) / 252
    gold_limit = max_gold_pct / 100.0

    def mdd_fn(w, r):
        port_rets = r @ w
        cum_rets = (1 + port_rets).cumprod()
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        return -drawdown.min()

    def ret_cons(w, r):
        port_rets = r @ w
        actual_cagr = ((1 + port_rets).cumprod().values[-1] ** (1/yrs)) - 1
        return 0.01 - abs(actual_cagr - target_ret)

    def gs_sum_cons(w, tickers):
        return gold_limit - sum(w[i] for i, t in enumerate(tickers) if t in ['GLD', 'SLV'])

    res = minimize(mdd_fn, [1./len(current_universe)]*len(current_universe), args=(rets,), 
                   bounds=[(0, 0.4)]*len(current_universe), 
                   constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1},
                                {'type':'ineq','fun':ret_cons,'args':(rets,)},
                                {'type':'ineq','fun':gs_sum_cons,'args':(current_universe,)}], method='SLSQP')
    
    if not res.success: return None, False
    top_idx = np.argsort(res.x)[-int(max_assets):]
    top_tickers = [current_universe[i] for i in top_idx]
    res_sub = minimize(mdd_fn, [1./len(top_tickers)]*len(top_tickers), args=(rets[top_tickers],),
                       bounds=[(0.05, 0.7)]*len(top_tickers), 
                       constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1},
                                    {'type':'ineq','fun':ret_cons,'args':(rets[top_tickers],)},
                                    {'type':'ineq','fun':gs_sum_cons,'args':(top_tickers,)}], method='SLSQP')
    
    final_res = res_sub if res_sub.success else res
    final_tickers = top_tickers if res_sub.success else current_universe
    weights = {final_tickers[i]: round(final_res.x[i]*100, 1) for i in range(len(final_tickers)) if final_res.x[i] > 0.01}
    return weights, not res_sub.success

# --- ETF 화면 실행 함수 ---
def run():
    st.title("⚖️ ETF 황금비율 설계소")
    
    if 'page' not in st.session_state: st.session_state.page = 'survey'
    if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
    if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0
    if 'max_assets' not in st.session_state: st.session_state.max_assets = 5
    if 'total_investment' not in st.session_state: st.session_state.total_investment = 10000.0
    if 'max_gold' not in st.session_state: st.session_state.max_gold = 10.0 

    if st.session_state.page == 'survey':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 30.0, float(st.session_state.target_return))
            st.session_state.target_mdd = st.number_input("허용 MDD (%)", 1.0, 50.0, float(st.session_state.target_mdd))
        with col2:
            st.session_state.max_assets = st.number_input("최대 종목 수", 3, 10, int(st.session_state.max_assets))
            st.session_state.max_gold = st.number_input("금/은 합산 한도 (%)", 0.0, 100.0, float(st.session_state.max_gold))
        with col3:
            st.session_state.total_investment = st.number_input("투자금 ($)", 100, 1000000, int(st.session_state.total_investment))
        if st.button("레시피 생성", use_container_width=True, type="primary"): 
            st.session_state.page = 'dashboard'; st.rerun()
    else:
        data = get_data(universe)
        wts, is_fb = find_robust_optimal(st.session_state.target_return, st.session_state.target_mdd, st.session_state.max_assets, st.session_state.max_gold, data)
        if wts:
            if st.button("⬅️ 다시 설계하기"): st.session_state.page = 'survey'; st.rerun()
            col_l, col_r = st.columns([1, 2.5])
            with col_l:
                st.subheader("💡 매수 가이드")
                for t, w in sorted(wts.items(), key=lambda x:x[1], reverse=True):
                    sec, desc = get_etf_details(t)
                    st.write(f"**{t}**: {w}% (${st.session_state.total_investment*w/100:,.0f})")
            with col_r:
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                st.plotly_chart(px.line(pd.DataFrame({"추천": pv, "SPY": norm['SPY']}), title="10년 성과"), use_container_width=True)
