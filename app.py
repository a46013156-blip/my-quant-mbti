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
if 'target_return' not in st.session_state: st.session_state.target_return = 15.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 12.0

# --- [유니버스 및 섹터 코멘트 정보] ---
ETF_INFO = {
    "지수/대형주": {"SPY": "미국 시장 대표 S&P500 지수 추종", "QQQ": "기술주 중심 나스닥100 지수 추종", "DIA": "우량주 중심 다우존스 지수 추종", "IWM": "중소형주 러셀2000 지수 추종"},
    "성장 섹터": {"SMH": "글로벌 반도체 기업 집중 투자", "XLK": "빅테크 및 IT 하드웨어/소프트웨어", "VGT": "정보 기술 전반적인 노출", "IBB": "바이오 테크 및 제약 혁신 기업", "LIT": "리튬 및 2차전지 밸류체인", "SKYY": "클라우드 컴퓨팅 테마"},
    "가치/배당 섹터": {"SCHD": "현금 흐름이 우수한 배당성장주", "VYM": "고배당 수익률 중심의 가치주", "VIG": "10년 이상 배당을 늘려온 우량주", "XLF": "은행 및 보험 등 금융 섹터", "XLE": "에너지 및 원유 관련 기업", "XLV": "전통적인 제약 및 헬스케어", "XLP": "식료품 등 경기 방어 성격의 필수소비재", "XLU": "안정적인 수익의 유틸리티", "XLRE": "부동산 임대 수익 중심의 리츠"},
    "투자 스타일": {"VUG": "이익 성장이 빠른 대형 성장주", "VTV": "저평가된 대형 가치주", "IJK": "중형 성장주", "IJJ": "중형 가치주"},
    "안전(국채)": {"SHV": "현금성 자산인 초단기 국채", "SHY": "변동성이 낮은 단기 국채", "IEF": "중간 정도의 듀레이션을 가진 중기 국채", "TLT": "금리 하락 시 수익이 극대화되는 장기 국채", "BIL": "초단기 현금 관리용"},
    "안전(기타)": {"LQD": "우량 기업이 발행한 투자등급 회사채", "HYG": "고수익을 추구하는 하이일드 채권", "TIP": "물가 상승에 연동되는 물가연동채", "BND": "미국 채권 시장 전체에 투자"},
    "대안자산": {"GLD": "대표적인 안전자산인 금 현물", "SLV": "산업용 수요와 귀금속 성격을 지닌 은", "USO": "국제 유가 변동에 투자", "DBA": "농산물 가격 변동에 투자", "UNG": "천연가스 가격 변동에 투자"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

def get_etf_details(ticker):
    for sector, etfs in ETF_INFO.items():
        if ticker in etfs:
            return sector, etfs[ticker]
    return "기타", "상세 정보 없음"

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
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.target_return = st.number_input("목표 수익률 (%)", 1.0, 40.0, float(st.session_state.target_return), 0.1)
    with col2:
        st.session_state.target_mdd = st.number_input("허용 MDD (%)", 1.0, 60.0, float(st.session_state.target_mdd), 0.1)
    
    if st.button("포트폴리오 생성 🚀", use_container_width=True, type="primary"):
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 퀀트 포트폴리오 리포트")
    
    # 🎯 입력 수치 요약
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32;">
        <b>🎯 설정된 투자 목표</b> | 연 수익률: <b>{st.session_state.target_return}%</b> | 허용 MDD: <b>-{st.session_state.target_mdd}%</b>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    if st.button("⬅️ 설정 수정"): st.session_state.page = 'survey'; st.rerun()

    with st.spinner('데이터 분석 및 최적 비중 계산 중...'):
        data = get_data(universe)
        wts = find_dual_optimal(st.session_state.target_return, st.session_state.target_mdd, data)
        
        if wts:
            col1, col2 = st.columns([1.3, 2.5])
            with col1:
                st.subheader("💡 추천 종목 및 섹터 코멘트")
                sorted_wts = sorted(wts.items(), key=lambda x: x[1], reverse=True)
                for t, w in sorted_wts:
                    sector, comment = get_etf_details(t)
                    st.markdown(f"**{t}** ({w}%)")
                    st.caption(f"**섹터:** {sector} | **코멘트:** {comment}")
                    st.write("---")
            
            with col2:
                # 성과 계산 및 그래프
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                df_plot = pd.DataFrame({"추천 포트폴리오": pv, "S&P 500 (SPY)": norm['SPY'], "나스닥 100 (QQQ)": norm['QQQ']})
                st.plotly_chart(px.line(df_plot, title="과거 10년 성과 비교"), use_container_width=True)
                
                # 📊 [요청사항 반영] CAGR, MDD 수치 비교표
                res_list = []
                for c in df_plot.columns:
                    yrs = len(df_plot[c])/252
                    cagr = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100
                    mdd = ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                    res_list.append({"자산 구분": c, "연평균 수익률(CAGR)": f"{cagr:.2f}%", "최대 낙폭(MDD)": f"{mdd:.2f}%"})
                
                st.subheader("📊 지수 대비 성과 요약")
                st.table(pd.DataFrame(res_list))
        else:
            st.error("⚠️ 해당 조건은 현재 데이터상 달성이 불가능합니다. 조건을 조정해 주세요.")
