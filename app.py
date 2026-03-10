
import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Quant MBTI Advisor", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'current_q' not in st.session_state: st.session_state.current_q = 0
if 'score' not in st.session_state: st.session_state.score = 0
if 'target_return' not in st.session_state: st.session_state.target_return = 10.0

# 자산군 구성 (개별종목 배제, ETF 중심)
ETF_INFO = {
    "채권": {"SHV": "초단기채", "IEF": "중기채", "TLT": "장기채"},
    "주식": {"SPY": "S&P500", "QQQ": "나스닥100", "SMH": "반도체", "LIT": "2차전지", "XLK": "기술주"},
    "원자재": {"GLD": "금", "USO": "원유"}
}
universe = [t for c in ETF_INFO.values() for t in c.keys()]

@st.cache_data(ttl=3600)
def get_data(tickers):
    # 비교를 위해 SPY와 QQQ는 무조건 포함해서 다운로드
    all_tickers = list(set(tickers + ['SPY', 'QQQ']))
    data = yf.download(all_tickers, period="10y", progress=False)['Close'].ffill().dropna()
    if data.index.tz is not None: data.index = data.index.tz_localize(None)
    return data

def optimize(target_pct, data, tickers):
    rets = data[tickers].pct_change().dropna()
    mu, cov = rets.mean() * 252, rets.cov() * 252
    target = target_pct / 100.0
    if target > mu.max(): return None, mu.max()
    
    # 리스크(분산) 최소화 함수
    def risk_fn(w): return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    
    res = minimize(risk_fn, [1./len(tickers)]*len(tickers),
                   bounds=[(0, 0.7)]*len(tickers), 
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w)-1}, 
                                {'type': 'eq', 'fun': lambda w: np.dot(w, mu)-target}])
    
    if not res.success: return None, None
    return {t: round(res.x[i]*100, 1) for i, t in enumerate(tickers) if res.x[i] > 0.01}

# --- 설문 화면 (1문항씩 출력) ---
if st.session_state.page == 'survey':
    questions = ["Q1. 하락장 대처?", "Q2. 목표 수익률?", "Q3. 정보 판단?", "Q4. 선호 수익?", "Q5. 관리 스타일?", "Q6. 레버리지?", "Q7. 분산 성향?"]
    options = [["손절", "방어", "추가매수"], ["5-7%", "8-12%", "15% 이상"], ["데이터", "스토리", "여론"], ["배당", "시세차익"], ["자동", "주기적", "실시간"], ["무시", "관심", "적극"], ["철저분산", "주식위주", "집중투자"]]
    
    st.title("🧭 나의 ETF 투자 성향 테스트")
    q_idx = st.session_state.current_q
    st.progress((q_idx + 1) / len(questions))
    st.subheader(f"{questions[q_idx]}")
    
    for idx, opt in enumerate(options[q_idx]):
        if st.button(opt, key=f"b_{q_idx}_{idx}", use_container_width=True):
            st.session_state.score += (idx + 1) * 3
            if q_idx < len(questions) - 1: st.session_state.current_q += 1
            else:
                s = st.session_state.score
                st.session_state.target_return = 18.0 if s >= 35 else (12.0 if s >= 20 else 7.0)
                st.session_state.page = 'dashboard'
            st.rerun()

# --- 결과 대시보드 ---
elif st.session_state.page == 'dashboard':
    st.title("🤖 글로벌 매크로 퀀트 대시보드")
    
    if st.button("🔄 다시 테스트하기"): 
        st.session_state.page = 'survey'; st.session_state.current_q = 0; st.session_state.score = 0; st.rerun()

    st.sidebar.header("⚙️ 시뮬레이션 설정")
    user_target = st.sidebar.number_input("목표 수익률 (%)", 5.0, 30.0, float(st.session_state.target_return), 1.0)

    data = get_data(universe)
    wts = optimize(user_target, data, universe)
    
    if wts:
        st.success(f"✅ 분석 완료! 설정된 목표 수익률: 연 {user_target}%")
        
        col1, col2 = st.columns([1, 2.2])
        with col1:
            st.subheader("💡 섹터별 최적 비중")
            for cat, etfs in ETF_INFO.items():
                cat_wts = {t: w for t, w in wts.items() if t in etfs}
                if cat_wts:
                    st.write(f"**[{cat}] : {sum(cat_wts.values()):.1f}%**")
                    for t, w in cat_wts.items(): st.write(f"  └ {t}: {w}%")
            st.info("※ 개별 종목을 배제한 자산배분 전략입니다.")

        with col2:
            # 성과 계산
            norm = (data / data.iloc[0]) * 100
            pv = sum([norm[t] * (w/100) for t, w in wts.items()])
            
            # 그래프용 데이터프레임 (SPY, QQQ 포함)
            df_plot = pd.DataFrame({
                "S&P 500 (SPY)": norm['SPY'],
                "Nasdaq 100 (QQQ)": norm['QQQ'],
                f"AI Portfolio (Target {user_target}%)": pv
            })
            
            st.plotly_chart(px.line(df_plot, title="10년 성과 비교 (지수 vs AI)"), use_container_width=True)

            # 성과 지표 계산 함수
            def get_metrics(series):
                yrs = len(series) / 252
                cagr = ((series.iloc[-1] / series.iloc[0])**(1/yrs)-1)*100
                mdd = ((series - series.cummax())/series.cummax()).min()*100
                return f"{cagr:.2f}%", f"{mdd:.2f}%"

            # 지표 표 생성
            metrics = []
            for col in df_plot.columns:
                c, m = get_metrics(df_plot[col])
                metrics.append({"비교 대상": col, "연평균 수익률 (CAGR)": c, "최대 낙폭 (MDD)": m})
            
            st.subheader("📊 벤치마크 대비 성과 요약")
            st.table(pd.DataFrame(metrics))
    else:
        st.error("해당 수익률은 물리적으로 불가능합니다. 목표를 낮춰주세요.")
