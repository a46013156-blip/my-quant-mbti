import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="ETF Recipe Builder", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'target_return' not in st.session_state: st.session_state.target_return = 12.0
if 'target_mdd' not in st.session_state: st.session_state.target_mdd = 15.0
if 'max_assets' not in st.session_state: st.session_state.max_assets = 5
if 'total_investment' not in st.session_state: st.session_state.total_investment = 10000.0
if 'max_gold' not in st.session_state: st.session_state.max_gold = 10.0 

TOLERANCE = 1.0 

ETF_INFO = {
    "지수/대형주": {"SPY": "S&P500 지수", "QQQ": "나스닥100 지수", "DIA": "다우존스 지수", "IWM": "러셀2000 지수"},
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
        if data.empty: return pd.DataFrame()
        valid_cols = data.columns[data.notna().sum() > (len(data) * 0.9)]
        if len(valid_cols) == 0: return pd.DataFrame()
        return data[valid_cols].ffill().dropna()
    except:
        return pd.DataFrame()

def find_robust_optimal(target_ret_pct, target_mdd_pct, max_assets, max_gold_pct, data):
    if data.empty or len(data) < 20: return None, False
    current_universe = [t for t in universe if t in data.columns]
    rets = data[current_universe].pct_change().dropna()
    if rets.empty or len(rets) < 2: return None, False
    
    target_ret, target_mdd = target_ret_pct / 100.0, target_mdd_pct / 100.0
    yrs = len(rets) / 252
    tol_val = TOLERANCE / 100.0
    gold_limit = max_gold_pct / 100.0

    def get_bounds(tickers, is_stage2=False):
        bnds = []
        for t in tickers:
            min_w = 0.05 if is_stage2 else 0.0
            max_w = 0.7 if is_stage2 else 0.4
            if min_w > max_w: min_w = max_w 
            bnds.append((min_w, max_w))
        return tuple(bnds)

    def mdd_fn(w, r):
        port_rets = r @ w
        if len(port_rets) == 0: return 1.0 
        cum_rets = (1 + port_rets).cumprod()
        drawdown = (cum_rets - cum_rets.cummax()) / cum_rets.cummax()
        return -drawdown.min()

    def ret_cons(w, r):
        port_rets = r @ w
        if len(port_rets) == 0: return -99.0 
        cum_ret_series = (1 + port_rets).cumprod()
        if len(cum_ret_series) == 0: return -99.0
        actual_cagr = (cum_ret_series.values[-1] ** (1/yrs)) - 1
        return tol_val - abs(actual_cagr - target_ret)

    def gold_silver_sum_cons(w, tickers):
        gs_weight = sum(w[i] for i, t in enumerate(tickers) if t in ['GLD', 'SLV'])
        return gold_limit - gs_weight

    cons1 = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': ret_cons, 'args': (rets,)},
        {'type': 'ineq', 'fun': gold_silver_sum_cons, 'args': (current_universe,)}
    ]

    bnds1 = get_bounds(current_universe, False)
    res = minimize(mdd_fn, [1./len(current_universe)]*len(current_universe), args=(rets,), 
                   bounds=bnds1, constraints=cons1, method='SLSQP')
    if not res.success: return None, False

    top_idx = np.argsort(res.x)[-int(max_assets):]
    top_tickers = [current_universe[i] for i in top_idx]
    rets_sub = rets[top_tickers]
    
    cons2 = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': ret_cons, 'args': (rets_sub,)},
        {'type': 'ineq', 'fun': gold_silver_sum_cons, 'args': (top_tickers,)}
    ]

    bnds2 = get_bounds(top_tickers, True)
    res_sub = minimize(mdd_fn, [1./len(top_tickers)]*len(top_tickers), args=(rets_sub,),
                       bounds=bnds2, constraints=cons2, method='SLSQP')

    is_fallback = not res_sub.success
    final_res = res_sub if res_sub.success else res
    final_tickers = top_tickers if res_sub.success else current_universe
    weights = {final_tickers[i]: round(final_res.x[i]*100, 1) for i in range(len(final_tickers)) if final_res.x[i] > 0.01}
    return weights, is_fallback

if st.session_state.page == 'survey':
    st.title("⚖️ 나만의 ETF 황금비율 설계소")
    st.write("10년 이상 검증된 데이터로 당신의 투자 레시피를 완성합니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.target_return = st.number_input("목표 연 수익률 (%)", 1.0, 30.0, float(st.session_state.target_return))
        st.session_state.target_mdd = st.number_input("허용 최대 하락률 (MDD, %)", 1.0, 50.0, float(st.session_state.target_mdd))
    with col2:
        st.session_state.max_assets = st.number_input("최대 구성 종목 수 (개)", 3, 10, int(st.session_state.max_assets))
        st.session_state.max_gold = st.number_input("금/은 합산 상한선 (%)", 0.0, 100.0, float(st.session_state.max_gold), help="금과 은의 합산 비중 한도를 설정합니다. 방어 자산이므로 10~15%를 권장합니다.")
    with col3:
        st.session_state.total_investment = st.number_input("총 투자 금액 (USD $)", 100, 1000000, int(st.session_state.total_investment))
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 시스템 초기화"):
            st.cache_data.clear(); st.rerun()
    
    if st.button("레시피 생성 🚀", use_container_width=True, type="primary"): 
        st.session_state.page = 'dashboard'; st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("🛡️ 나만의 AI 포트폴리오 처방전")
    st.markdown(f"""
    <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; border-left: 5px solid #1e293b; margin-bottom: 20px;">
        💰 총 투자액: <b>${st.session_state.total_investment:,.0f}</b> | 금/은 합산 제한: <b>최대 {st.session_state.max_gold}%</b><br>
        목표 수익률: <b>{st.session_state.target_return}%</b> | 목표 MDD: <b>-{st.session_state.target_mdd}%</b>
    </div>
    """, unsafe_allow_html=True)

    data = get_data(universe)
    if data.empty or len(data) < 20:
        st.error("⚠️ 금융 데이터를 불러오는 데 실패했습니다. 뒤로 가서 '시스템 초기화' 버튼을 눌러주세요.")
        if st.button("⬅️ 돌아가기"): st.session_state.page = 'survey'; st.rerun()
    else:
        wts, is_fallback = find_robust_optimal(st.session_state.target_return, st.session_state.target_mdd, st.session_state.max_assets, st.session_state.max_gold, data)
        if wts:
            if is_fallback: st.warning("⚠️ 입력하신 종목 수 내에서는 완벽한 조준이 어려워 전체 최적 비중을 제안합니다.")
            if st.button("⬅️ 다시 설계하기"): st.session_state.page = 'survey'; st.rerun()

            col_l, col_r = st.columns([1.2, 2.5])
            with col_l:
                st.subheader("💡 매수 가이드")
                sorted_wts = sorted(wts.items(), key=lambda x: x[1], reverse=True)
                for t, w in sorted_wts:
                    sec, desc = get_etf_details(t)
                    amt = (st.session_state.total_investment * w) / 100
                    st.markdown(f"""<div style='padding:10px; background:white; border:1px solid #e2e8f0; border-radius:6px; margin-bottom:6px;'>
                        <div style='display:flex; justify-content:space-between;'><b>{t}</b> <b style='color:#16a34a;'>${amt:,.0f} ({w}%)</b></div>
                        <div style='font-size:0.75rem; color:#64748b;'>{sec} · {desc}</div></div>""", unsafe_allow_html=True)
            with col_r:
                norm = (data / data.iloc[0]) * 100
                pv = sum([norm[t] * (w/100) for t, w in wts.items()])
                df_p = pd.DataFrame({"추천": pv, "S&P 500": norm['SPY'], "나스닥 100": norm['QQQ']})
                st.plotly_chart(px.line(df_p, title="과거 10년 성과 비교 시뮬레이션"), use_container_width=True)
                res_list = []
                for c in df_p.columns:
                    y = len(df_p[c])/252
                    cagr, mdd = ((df_p[c].values[-1]/df_p[c].values[0])**(1/y)-1)*100, ((df_p[c]-df_p[c].cummax())/df_p[c].cummax()).min()*100
                    res_list.append({"구분": c, "CAGR": f"{cagr:.2f}%", "MDD": f"{mdd:.2f}%"})
                st.table(pd.DataFrame(res_list))
        else:
            st.error("⚠️ 조건을 만족하는 레시피를 찾을 수 없습니다. 수익률을 낮추거나 MDD 한도를 늘려보세요.")
            if st.button("⬅️ 다시 조준하기"): st.session_state.page = 'survey'; st.rerun()
