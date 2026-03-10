import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Precision Quant Advisor", layout="wide")

# --- 세션 및 데이터 설정 (생략 방지 위해 핵심만 포함) ---
if 'page' not in st.session_state: st.session_state.page = 'survey'
ETF_INFO = {
    "지수/성장": {"SPY": "S&P500", "QQQ": "나스닥100", "SMH": "반도체", "XLK": "기술주"},
    "가치/배당": {"SCHD": "배당성장", "VYM": "고배당", "XLF": "금융", "XLV": "헬스케어"},
    "채권/안전": {"IEF": "중기채", "TLT": "장기채", "SHV": "초단기채", "GLD": "금"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    data = yf.download(tickers + ['SPY', 'QQQ'], period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_precise_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret = target_ret_pct / 100.0
    max_mdd_allowed = target_mdd_pct / 100.0
    yrs = len(rets) / 252

    # 목적함수: 변동성 최소화
    def vol_fn(w): return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))

    # 🌟 제약 조건: 목표 수익률의 ±0.5% 이내로 근접하도록 설정
    def ret_range_cons(w):
        cum_ret = (1 + (rets @ w)).cumprod().iloc[-1]
        cagr = (cum_ret ** (1/yrs)) - 1
        return 0.005 - abs(cagr - target_ret) # 0.5% 오차범위

    # 🌟 제약 조건: MDD는 목표치보다 더 낮아야 함 (방어)
    def mdd_cons(w):
        cum_rets = (1 + (rets @ w)).cumprod()
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        return max_mdd_allowed - (-drawdown.min())

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': ret_range_cons},
            {'type': 'ineq', 'fun': mdd_cons}]
    
    res = minimize(vol_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.5)]*len(universe), constraints=cons, method='SLSQP')
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- UI 부분 ---
if st.session_state.page == 'survey':
    st.title("🎯 정밀 목표 설정")
    st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 30.0, 15.0, 0.1)
    st.session_state.target_mdd = st.number_input("허용 MDD (%)", 1.0, 50.0, 12.0, 0.1)
    if st.button("포트폴리오 생성 🚀"):
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("📊 성과 분석 및 목표 달성도")
    data = get_data(universe)
    wts = find_precise_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
    
    if wts:
        # 성과 계산
        norm = (data / data.iloc[0]) * 100
        pv = sum([norm[t] * (w/100) for t, w in wts.items()])
        
        yrs = len(pv)/252
        actual_cagr = ((pv.iloc[-1]/pv.iloc[0])**(1/yrs)-1)*100
        actual_mdd = ((pv - pv.cummax())/pv.cummax()).min()*100

        # 🌟 [신규] 목표 대비 실제 달성 수치 비교 (오차 확인)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("CAGR 달성도", f"{actual_cagr:.2f}%", f"{actual_cagr - st.session_state.target_return:+.2f}% (목표 대비)")
        col_m2.metric("MDD 방어력", f"{actual_mdd:.2f}%", f"{abs(actual_mdd) - st.session_state.target_mdd:+.2f}% (한도 대비)", delta_color="inverse")
        
        st.write("---")
        # 기존 차트 및 표 출력 코드...
        df_plot = pd.DataFrame({"Portfolio": pv, "SPY": norm['SPY'], "QQQ": norm['QQQ']})
        st.plotly_chart(px.line(df_plot, title="과거 성과 시뮬레이션"), use_container_width=True)
        
        # 섹터 정보 포함 비중
        st.subheader("💡 추천 포트폴리오 비중")
        for t, w in wts.items():
            st.write(f"**{t}**: {w}%")
    else:
        st.error("⚠️ 입력하신 조건은 수학적 오차범위(±0.5%) 내에서도 해를 찾을 수 없습니다. 조건을 완화해 주세요.")
