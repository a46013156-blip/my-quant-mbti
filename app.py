import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime

st.set_page_config(page_title="Advanced Quant Advisor", layout="wide")

# --- [고도화된 질문 및 점수 체계] ---
if 'page' not in st.session_state: st.session_state.page = 'survey'
if 'current_q' not in st.session_state: st.session_state.current_q = 0
if 'score' not in st.session_state: st.session_state.score = 0

def update_app():
    # 1. 투자 기간, 2. 손실 감내도, 3. 지식 수준, 4. 하락장 반응 등
    questions = [
        "Q1. 이 자금을 얼마나 오랫동안 투자하실 계획인가요?",
        "Q2. 투자 원금의 얼마까지 손실을 감내하실 수 있나요?",
        "Q3. 금융 시장의 변동성에 대해 얼마나 잘 이해하고 계신가요?",
        "Q4. 과거에 큰 하락장을 겪었을 때 당신의 행동은 어떠했나요?"
    ]
    options = [
        ["1년 미만 (매우 단기)", "1~3년 (중기)", "5년 이상 (장기)"],
        ["원금 절대 보존 (-0%)", "최대 -15% 수준", "최대 -30% 이상도 가능"],
        ["초보자 (예적금 위주)", "중급자 (주식/ETF 유경험)", "전문가 (파생/퀀트 이해)"],
        ["공포에 질려 매도", "덤덤하게 유지", "기회로 보고 추가 매수"]
    ]
    # 점수 가중치 (100점 만점 설계)
    weights = [10, 25, 15, 50] 

    if st.session_state.page == 'survey':
        st.title("🧭 투자 성향 정밀 진단")
        q_idx = st.session_state.current_q
        st.progress((q_idx + 1) / len(questions))
        st.subheader(questions[q_idx])

        for i, opt in enumerate(options[q_idx]):
            if st.button(opt, key=f"q_{q_idx}_{i}", use_container_width=True):
                # 점수 계산 로직 (가중치 적용)
                st.session_state.score += (i * weights[q_idx])
                if q_idx < len(questions) - 1:
                    st.session_state.current_q += 1
                else:
                    # 최종 스코어 기반 목표 수익률 매핑
                    s = st.session_state.score
                    if s >= 150: st.session_state.target_return = 15.0
                    elif s >= 80: st.session_state.target_return = 10.0
                    else: st.session_state.target_return = 6.0
                    st.session_state.page = 'dashboard'
                st.rerun()

    elif st.session_state.page == 'dashboard':
        st.title("🤖 AI 최적화 포트폴리오 리포트")
        # (기존 최적화 및 그래프 코드 생략 - 동일하게 유지하되 UI만 개선)
        st.write(f"당신의 투자 스코어: {st.session_state.score}점")
        st.success(f"최적 목표 수익률: 연 {st.session_state.target_return}%")
        # ... 후속 대시보드 로직 ...

update_app()
