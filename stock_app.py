import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run():
    st.title("📈 개별종목 딥다이브 분석")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # 기본값을 AAPL(애플)로 세팅해 둡니다.
        ticker = st.text_input("🔍 분석할 종목의 티커를 입력하세요 (예: AAPL, TSLA, MSFT)", "AAPL").upper()
    
    if ticker:
        try:
            with st.spinner(f"'{ticker}' 데이터를 불러오는 중입니다..."):
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")
            
            if hist.empty:
                st.error("데이터를 불러올 수 없습니다. 티커를 확인해주세요.")
                return
            
            # 🌟 1. 상단 핵심 요약 지표 (Metrics)
            st.subheader(f"{info.get('longName', ticker)} ({ticker})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재가", f"${info.get('currentPrice', hist['Close'].iloc[-1]):.2f}")
            c2.metric("52주 최고가", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            
            # 시가총액을 보기 쉽게 Billion(10억) 단위로 변환
            mcap = info.get('marketCap', 0)
            c3.metric("시가총액", f"${mcap / 1000000000:.2f}B" if mcap else "N/A")
            c4.metric("PER (주가수익비율)", f"{info.get('trailingPE', 'N/A')}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 🌟 2. 이동평균선 계산 (20일, 60일)
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            
            # 🌟 3. 전문가용 캔들스틱 + 거래량 분할 차트 (Plotly)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f"최근 1년 주가 흐름", "거래량"), 
                                row_width=[0.2, 0.7])
            
            # 캔들차트 추가
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="주가"), row=1, col=1)
            
            # 이동평균선 추가
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1.5), name="20일 이평선"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='blue', width=1.5), name="60일 이평선"), row=1, col=1)
            
            # 거래량 차트 추가 (상승=초록, 하락=빨강)
            colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for index, row in hist.iterrows()]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name="거래량"), row=2, col=1)
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: 입력하신 티커가 정확한지 확인해주세요.")
