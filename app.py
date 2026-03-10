import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

st.set_page_config(page_title="Advanced Quant Advisor", layout="wide")

# --- 세션 관리 ---
if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'current_q' not in st.session_state: st.session_state.current_q = 0
if 'score' not in st.session_state: st.session_state.score = 0
if 'target_return' not in st.session_state: st.session_state.target_return = 10.0

# --- 투자 유니버스 (ETF 구성) ---
ETF_INFO = {
    "주식 (Stocks)": {"SPY": "S&P500", "QQQ": "나스닥100", "SMH": "반도체", "XLK": "기술주", "XLV": "헬스케어"},
    "채권 (Bonds)": {"IEF": "중기채", "TLT": "장기채", "SHV": "초단기채"},
    "대안자산 (Alt)": {"GLD": "금", "USO": "원유"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]
ticker_names = {t: n for c in ETF_INFO.values() for t, n in c.items()}

@st.cache_data(ttl=3600)
def get_data(tickers):
    # 벤치마크용 지수 포함 10년치 데이터 다운로드
    data = yf.download(tickers + ['SPY', 'QQQ'], period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def optimize(target_pct, data):
    rets = data[universe].pct_change().dropna()
    mu, cov = rets.mean() * 252, rets.cov() * 252
    target = target_pct / 100.0
    
    if target > mu.max(): return None
    
    # 목적함수: 포트폴리오 분산(위험) 최소화
    def risk_fn(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    
    res = minimize(risk_fn, [1./len(universe)]*len(universe),
                   bounds=[(0, 0.7)]*len(universe), # 한 종목 최대 70% 제한
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, 
                                {'type': 'eq', 'fun': lambda w: np.dot(w, mu)-target}])
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# ==========================================
# 화면 1: 설문조사
# ==========================================
if st.session_state.page == 'survey':
    st.title("🧭 투자 성향 정밀 진단")
    questions = ["Q1. 투자 기간?", "Q2. 감내 가능 손실?", "Q3. 금융 지식?", "Q4. 폭락장 반응?"]
    options = [["1년 미만", "1-3년", "5년 이상"], ["원금보존", "-15%", "-30%+"], ["초보", "중급", "전문가"], ["매도", "유지", "추가매수"]]
    weights = [10, 25, 15, 50] 
    
    q_idx = st.session_state.current_q
    st.progress((q_idx + 1) / len(questions))
    st.subheader(questions[q_idx])

    for i, opt in enumerate(options[q_idx]):
        if st.button(opt, key=f"q_{q_idx}_{i}", use_container_width=True):
            st.session_state.score += (i * weights[q_idx])
            if q_idx < len(questions) - 1:
                st.session_state.current_q += 1
                st.rerun()
            else:
                s = st.session_state.score
                # 스코어에 따른 목표 수익률 매핑
                if s >= 150: st.session_state.target_return = 15.0
                elif s >= 80: st.session_state.target_return = 11.0
                else: st.session_state.target_return = 7.0
                st.session_state.page = 'dashboard'
                st.rerun()

# ==========================================
# 화면 2: 결과 대시보드 (처방전 출력)
# ==========================================
elif st.session_state.page == 'dashboard':
    st.title("🤖 AI 최적화 포트폴리오 리포트")
    st.write(f"당신의 투자 스코어: **{st.session_state.score}점**")
    st.success(f"최적 목표 수익률: **연 {st.session_state.target_return}%**")
    
    if st.button("🔄 테스트 다시 하기"):
        st.session_state.page = 'survey'; st.session_state.current_q = 0; st.session_state.score = 0; st.rerun()

    with st.spinner('실시간 금융 데이터 분석 및 최적화 중...'):
        data = get_data(universe)
        wts = optimize(st.session_state.target_return, data)
        
        if wts:
            col1, col2 = st.columns([1, 2.2])
            with col1:
                st.subheader("💡 최적 자산 비중")
                for cat, etfs in ETF_INFO.items():
                    cat_wts = {t: w for t, w in wts.items() if t in etfs}
                    if cat_wts:
                        st.markdown(f"**[{cat}] : {sum(cat_wts.values()):.1f}%**")
                        for t, w in cat_wts.items():
                            st.write(f"&nbsp;&nbsp;└ {t} ({ticker_names[t]}): {w}%")
                        st.write("")
            
            with col2:
                # 성과 계산
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                
                df_plot = pd.DataFrame({
                    "AI Portfolio": pv,
                    "S&P 500 (SPY)": norm['SPY'],
                    "Nasdaq 100 (QQQ)": norm['QQQ']
                })
                
                st.plotly_chart(px.line(df_plot, title="과거 10년 시뮬레이션 (지수 대비 성과)"), use_container_width=True)
                
                # 지표 요약
                metrics = []
                for col in df_plot.columns:
                    yrs = len(df_plot[col]) / 252
                    cagr = ((df_plot[col].iloc[-1] / df_plot[col].iloc[0])**(1/yrs)-1)*100
                    mdd = ((df_plot[col] - df_plot[col].cummax())/df_plot[col].cummax()).min()*100
                    metrics.append({"대상": col, "수익률(CAGR)": f"{cagr:.2f}%", "위험도(MDD)": f"{mdd:.2f}%"})
                st.table(pd.DataFrame(metrics))
        else:
            st.error("현재 시장 상황에서 해당 수익률 달성은 수학적으로 불가능합니다. 수치를 낮춰주세요.")
