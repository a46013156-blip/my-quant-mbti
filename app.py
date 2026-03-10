import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Mega-Universe Quant", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0

# --- [유니버스 대폭 확장: 60개 핵심 ETF] ---
ETF_INFO = {
    "지수/대형": {"SPY": "S&P500", "QQQ": "나스닥100", "DIA": "다우존스", "IWM": "러셀2000"},
    "섹터(성장)": {"SMH": "반도체", "XLK": "기술주", "VGT": "IT전체", "IBB": "바이오", "LIT": "2차전지", "SKYY": "클라우드"},
    "섹터(가치/배당)": {"SCHD": "배당성장", "VYM": "고배당", "VIG": "배당귀족", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어", "XLP": "필수소비재", "XLU": "유틸리티", "XLRE": "리츠"},
    "스타일": {"VUG": "대형성장", "VTV": "대형가치", "IJK": "중형성장", "IJJ": "중형가치"},
    "채권(국채)": {"SHV": "초단기채", "SHY": "단기채", "IEF": "중기채", "TLT": "장기채", "BIL": "1-3개월채"},
    "채권(회사/기타)": {"LQD": "우량회사채", "HYG": "하이일드", "TIP": "물가연동채", "BND": "채권전체"},
    "대안자산": {"GLD": "금", "SLV": "은", "USO": "원유", "DBA": "농산물", "UNG": "천연가스"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    # 유니버스가 커졌으므로 효율적인 다운로드 필요
    data = yf.download(tickers, period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_dual_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    target_ret = target_ret_pct / 100.0
    max_mdd_allowed = target_mdd_pct / 100.0
    yrs = len(rets) / 252

    def vol_fn(w): return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))
    
    def ret_cons(w):
        cum_ret = (1 + (rets @ w)).cumprod().iloc[-1]
        return ((cum_ret ** (1/yrs)) - 1) - target_ret

    def mdd_cons(w):
        cum_rets = (1 + (rets @ w)).cumprod()
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        return max_mdd_allowed - (-drawdown.min())

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': ret_cons},
            {'type': 'ineq', 'fun': mdd_cons}]
    
    # 종목수가 많아졌으므로 종목당 비중 제한을 30%로 낮춰 분산 효과 극대화
    res = minimize(vol_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.3)]*len(universe), constraints=cons, method='SLSQP')
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- UI 레이아웃 ---
if st.session_state.page == 'survey':
    st.title("🏛️ 메가 유니버스 퀀트 엔진")
    st.write("미국 상장 대표 ETF 60개를 분석하여 당신의 목표에 최적화된 조합을 찾습니다.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 40.0, float(st.session_state.target_return), 0.1)
    with col2:
        st.session_state.target_mdd = st.number_input("허용 MDD (%)", 1.0, 60.0, float(st.session_state.target_mdd), 0.1)
    
    if st.button("60개 ETF 전수 조사 및 최적화 실행 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 60개 ETF 최적 자산배분 결과")
    if st.button("⬅️ 설정 수정"): st.session_state.page = 'survey'; st.rerun()

    with st.spinner('방대한 데이터를 기반으로 최적의 조합을 연산 중입니다...'):
        data = get_data(universe)
        wts = find_dual_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
        
        if wts:
            col1, col2 = st.columns([1, 2.5])
            with col1:
                st.subheader("💡 최적 비중")
                for t, w in sorted(wts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{t}**: {w}%")
            with col2:
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                df_plot = pd.DataFrame({"Portfolio": pv, "SPY": norm['SPY'], "QQQ": norm['QQQ']})
                st.plotly_chart(px.line(df_plot, title="10년 성과 비교 (Benchmark vs AI)"))
                
                # 성과 표
                res_list = []
                for c in df_plot.columns:
                    yrs = len(df_plot[c])/252
                    cagr = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100
                    mdd = ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                    res_list.append({"구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
                st.table(pd.DataFrame(res_list))
        else:
            st.error("⚠️ 60개의 ETF로도 해당 조건(수익률/MDD)을 만족할 수 없습니다. 목표를 현실적으로 조정해주세요.")
