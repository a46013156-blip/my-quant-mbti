import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Concentrated Quant Advisor", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0
if 'max_assets' not in st.session_state: st.session_state.max_assets = 5
TOLERANCE = 1.0 

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

def find_robust_optimal(target_ret_pct, target_mdd_pct, max_assets, data):
    rets = data[universe].pct_change().dropna()
    target_ret, target_mdd = target_ret_pct / 100.0, target_mdd_pct / 100.0
    yrs, tol_val = len(rets) / 252, TOLERANCE / 100.0

    def mdd_fn(w, r):
        cum_rets = (1 + (r @ w)).cumprod()
        return -((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()

    def ret_cons(w, r):
        cagr = ((1 + (r @ w)).cumprod().iloc[-1] ** (1/yrs)) - 1
        return tol_val - abs(cagr - target_ret)

    # 🌟 1단계: 전체 유니버스에서 최적화 시도
    res = minimize(mdd_fn, [1./len(universe)]*len(universe), args=(rets,), 
                   bounds=[(0, 0.5)]*len(universe), 
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                {'type': 'ineq', 'fun': ret_cons, 'args': (rets,)}], method='SLSQP')

    if not res.success: return None, False

    # 🌟 2단계: 종목 수 제한 (상위 N개 추출 후 재최적화)
    top_idx = np.argsort(res.x)[-int(max_assets):]
    top_tickers = [universe[i] for i in top_idx]
    rets_sub = rets[top_tickers]
    
    res_sub = minimize(mdd_fn, [1./len(top_tickers)]*len(top_tickers), args=(rets_sub,),
                       bounds=[(0.05, 0.6)]*len(top_tickers), # 최소 5% 이상 담도록 설정
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'ineq', 'fun': ret_cons, 'args': (rets_sub,)}], method='SLSQP')

    is_fallback = not res_sub.success
    final_res = res_sub if res_sub.success else res # 실패 시 전체 비중 반환
    final_tickers = top_tickers if res_sub.success else universe
    
    weights = {final_tickers[i]: round(final_res.x[i]*100, 1) for i in range(len(final_tickers)) if final_res.x[i] > 0.01}
    return weights, is_fallback

# --- UI ---
if st.session_state.page == 'survey':
    st.title("🏛️ 압축형 정밀 퀀트 엔진")
    col1, col2, col3 = st.columns(3)
    st.session_state.target_return = col1.number_input("목표 수익률 (%)", 1.0, 30.0, float(st.session_state.target_return), 0.1)
    st.session_state.target_mdd = col2.number_input("목표 MDD (%)", 1.0, 50.0, float(st.session_state.target_mdd), 0.1)
    st.session_state.max_assets = col3.number_input("최대 종목 수 (개)", 3, 15, int(st.session_state.max_assets))
    
    if st.button("핵심 포트폴리오 생성 🚀", use_container_width=True, type="primary"): 
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 포트폴리오 분석 결과")
    st.markdown(f"""
    <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border-left: 5px solid #1e293b; margin-bottom: 20px;">
        <span style="font-size: 0.9rem; color: #64748b;">🎯 타겟 설정: {st.session_state.max_assets}개 종목 이내 (오차 ±{TOLERANCE}%)</span><br>
        <b style="font-size: 1.1rem;">수익률 {st.session_state.target_return}% | 목표 MDD -{st.session_state.target_mdd}%</b>
    </div>
    """, unsafe_allow_html=True)

    data = get_data(universe)
    wts, is_fallback = find_robust_optimal(st.session_state.target_return, st.session_state.target_mdd, st.session_state.max_assets, data)

    if wts:
        if is_fallback: st.warning("⚠️ 입력하신 종목 수 제한 내에서는 해를 찾기 어려워 전체 유니버스 비중을 보여드립니다.")
        else: st.success(f"✅ 핵심 {len(wts)}개 종목으로 구성된 최적의 조합을 찾았습니다.")

        if st.button("⬅️ 설정 수정"): st.session_state.page = 'survey'; st.rerun()

        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("💡 추천 비중")
            sorted_wts = sorted(wts.items(), key=lambda x: x[1], reverse=True)
            for t, w in sorted_wts:
                sector, comment = get_etf_details(t)
                st.markdown(f"""<div style="padding: 6px 10px; background-color: white; border: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 1rem; font-weight: 700; color: #0f172a;">{t}</span>
                        <span style="font-size: 1rem; font-weight: 700; color: #16a34a;">{w}%</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; line-height: 1.2; margin-top: 2px;"><b>{sector}</b> · {comment}</div>
                </div>""", unsafe_allow_html=True)
        
        with col2:
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            df_plot = pd.DataFrame({"추천 포트폴리오": pv, "S&P 500": norm['SPY'], "나스닥 100": norm['QQQ']})
            st.plotly_chart(px.line(df_plot, title="과거 10년 성과 시뮬레이션"), use_container_width=True)
            
            res_list = []
            for c in df_plot.columns:
                yrs = len(df_plot[c])/252
                cagr, mdd = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100, ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                res_list.append({"자산 구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
            st.table(pd.DataFrame(res_list))
