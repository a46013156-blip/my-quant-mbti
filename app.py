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

# --- [유니버스 및 섹터 정보] ---
ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500", "QQQ": "나스닥100", "DIA": "다우존스", "IWM": "러셀2000"},
    "성장 섹터": {"SMH": "반도체", "XLK": "기술주", "VGT": "IT전체", "IBB": "바이오", "LIT": "2차전지", "SKYY": "클라우드"},
    "가치/배당 섹터": {"SCHD": "배당성장", "VYM": "고배당", "VIG": "배당귀족", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어", "XLP": "필수소비재", "XLU": "유틸리티", "XLRE": "리츠"},
    "투자 스타일": {"VUG": "대형성장", "VTV": "대형가치", "IJK": "중형성장", "IJJ": "중형가치"},
    "안전자산(국채)": {"SHV": "초단기채", "SHY": "단기채", "IEF": "중기채", "TLT": "장기채", "BIL": "1-3개월채"},
    "안전자산(기타)": {"LQD": "우량회사채", "HYG": "하이일드", "TIP": "물가연동채", "BND": "채권전체"},
    "인플레/대안자산": {"GLD": "금", "SLV": "은", "USO": "원유", "DBA": "농산물", "UNG": "천연가스"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

# 티커별 섹터 및 이름 매핑 헬퍼 함수
def get_etf_details(ticker):
    for sector, etfs in ETF_INFO.items():
        if ticker in etfs:
            return sector, etfs[ticker]
    return "기타", "알 수 없음"

@st.cache_data(ttl=3600)
def get_data(tickers):
    all_tickers = list(set(tickers + ['SPY', 'QQQ']))
    data = yf.download(all_tickers, period="10y", progress=False)['Close'].ffill().dropna()
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
    
    res = minimize(vol_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.4)]*len(universe), constraints=cons, method='SLSQP')
    
    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- UI 레이아웃 ---
if st.session_state.page == 'survey':
    st.title("🏛️ 메가 유니버스 퀀트 엔진")
    st.write("원하는 수치를 입력하고 최적화 버튼을 눌러주세요.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 40.0, float(st.session_state.target_return), 0.1)
    with col2:
        st.session_state.target_mdd = st.number_input("허용 MDD (%)", 1.0, 60.0, float(st.session_state.target_mdd), 0.1)
    
    if st.button("포트폴리오 생성 시작 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 퀀트 포트폴리오 리포트")
    
    # 🌟 [요청사항 반영] 내가 입력한 수치 강조 표시
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
        <h4 style="margin: 0;">🎯 설정하신 투자 목표</h4>
        <p style="font-size: 18px; margin: 10px 0 0 0;">
            목표 수익률: <b>연 {st.session_state.target_return}%</b> | 
            허용 최대 낙폭(MDD): <b>-{st.session_state.target_mdd}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    if st.button("⬅️ 설정 수정하러 가기"): st.session_state.page = 'survey'; st.rerun()

    with st.spinner('방대한 데이터를 기반으로 최적 비중을 계산 중입니다...'):
        data = get_data(universe)
        wts = find_dual_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
        
        if wts:
            col1, col2 = st.columns([1.2, 2.5])
            with col1:
                st.subheader("💡 섹터별 추천 비중")
                # 🌟 [요청사항 반영] 섹터 정보를 포함한 비중 출력
                sorted_wts = sorted(wts.items(), key=lambda x: x[1], reverse=True)
                for t, w in sorted_wts:
                    sector, name = get_etf_details(t)
                    st.markdown(f"""
                    **{t}** <span style='color: gray; font-size: 0.8em;'>[{sector}]</span>  
                    {name} : **{w}%**
                    """, unsafe_allow_html=True)
                    st.write("---")
            
            with col2:
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                df_plot = pd.DataFrame({"Portfolio": pv, "SPY": norm['SPY'], "QQQ": norm['QQQ']})
                st.plotly_chart(px.line(df_plot, title="10년 성과 비교 (Benchmark vs AI)"), use_container_width=True)
                
                # 성과 지표 표
                res_list = []
                for c in df_plot.columns:
                    yrs = len(df_plot[c])/252
                    cagr = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100
                    mdd = ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                    res_list.append({"구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
                st.table(pd.DataFrame(res_list))
        else:
            st.error("⚠️ 해당 조건은 현재 데이터상 달성이 불가능합니다. 수익률을 낮추거나 MDD 한도를 늘려주세요.")
