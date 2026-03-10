import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Target Tracking Quant", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 10.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 12.0

ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500", "QQQ": "나스닥100", "DIA": "다우존스", "IWM": "러셀2000"},
    "성장 섹터": {"SMH": "반도체", "XLK": "기술주", "VGT": "IT전체", "IBB": "바이오", "LIT": "2차전지"},
    "가치/배당": {"SCHD": "배당성장", "VYM": "고배당", "VIG": "배당귀족", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어"},
    "안전자산": {"SHV": "초단기국채", "IEF": "중기국채", "TLT": "장기국채", "GLD": "금", "SLV": "은", "BND": "종합채권"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    data = yf.download(list(set(tickers + ['SPY', 'QQQ'])), period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def find_tracking_optimal(target_ret_pct, target_mdd_pct, data):
    rets = data[universe].pct_change().dropna()
    t_ret, t_mdd = target_ret_pct / 100.0, target_mdd_pct / 100.0
    yrs = len(rets) / 252

    # 🌟 목적함수: 목표 수익률과 목표 MDD로부터의 '거리'를 최소화
    def distance_fn(w):
        port_rets = rets @ w
        cum_rets = (1 + port_rets).cumprod()
        
        # 실제 CAGR 계산
        actual_cagr = (cum_rets.iloc[-1] ** (1/yrs)) - 1
        # 실제 MDD 계산
        actual_mdd = -((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min()
        
        # 두 오차의 제곱합 (수익률 오차에 가중치 2배 부여하여 더 강력하게 추적)
        return ((actual_cagr - t_ret)**2 * 2) + (actual_mdd - t_mdd)**2

    res = minimize(distance_fn, [1./len(universe)]*len(universe), 
                   bounds=[(0, 0.4)]*len(universe), 
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                   method='SLSQP')

    if not res.success: return None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(universe) if res.x[i] > 0.01}

# --- UI ---
if st.session_state.page == 'survey':
    st.title("🎯 정밀 타겟 조준 엔진")
    st.write("입력하신 수익률과 MDD 수치에 **가장 가까운** 조합을 찾아냅니다.")
    col1, col2 = st.columns(2)
    st.session_state.target_return = col1.number_input("목표 수익률 (%)", 1.0, 30.0, float(st.session_state.target_return), 0.1)
    st.session_state.target_mdd = col2.number_input("목표 MDD (%)", 1.0, 50.0, float(st.session_state.target_mdd), 0.1)
    if st.button("정밀 추적 시작 🚀", use_container_width=True, type="primary"): 
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 정밀 추적 결과 리포트")
    st.info(f"조준 목표 👉 수익률: {st.session_state.target_return}% | MDD: -{st.session_state.target_mdd}%")
    
    data = get_data(universe)
    wts = find_tracking_optimal(st.session_state.target_return, st.session_state.target_mdd, data)

    if wts:
        if st.button("⬅️ 다시 조준하기"): st.session_state.page = 'survey'; st.rerun()
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("💡 추천 비중")
            for t, w in sorted(wts.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"**{t}**: {w}%")
        with col2:
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            df_plot = pd.DataFrame({"Portfolio": pv, "SPY": norm['SPY'], "QQQ": norm['QQQ']})
            st.plotly_chart(px.line(df_plot, title="과거 성과 시뮬레이션"), use_container_width=True)
            
            res_list = []
            for c in df_plot.columns:
                yrs = len(df_plot[c])/252
                cagr, mdd = ((df_plot[c].iloc[-1]/df_plot[c].iloc[0])**(1/yrs)-1)*100, ((df_plot[c]-df_plot[c].cummax())/df_plot[c].cummax()).min()*100
                res_list.append({"자산 구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
            st.table(pd.DataFrame(res_list))
    else:
        st.error("해당 지점을 조준할 수 없습니다. 목표를 변경해주세요.")
