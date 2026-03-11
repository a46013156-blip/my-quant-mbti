import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def run():
    st.title("📈 개별종목 딥다이브 분석")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("🔍 분석할 종목의 티커를 입력하세요 (예: AAPL, TSLA, NVDA)", "NVDA").upper()
    
    if ticker:
        try:
            with st.spinner(f"'{ticker}' 데이터를 불러오는 중입니다..."):
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
            
            if hist.empty:
                st.warning("⚠️ 주가 데이터를 불러올 수 없습니다. 티커명을 확인해주세요.")
                return
            
            # --- 🛡️ 야후 파이낸스 정보 차단 우회 로직 ---
            info = {}
            fast_info = {}
            try: info = stock.info
            except: pass
            
            try: fast_info = stock.fast_info
            except: pass

            # 🌟 1. 상단 핵심 요약 지표 (Metrics)
            name = info.get('longName', ticker)
            st.subheader(f"{name} ({ticker})")
            
            c1, c2, c3, c4 = st.columns(4)
            
            current_price = hist['Close'].iloc[-1]
            c1.metric("현재가", f"${current_price:.2f}")
            
            high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
            c2.metric("52주 최고가", f"${high_52w:.2f}")
            
            # 시가총액 (fast_info 우선 적용으로 차단 우회)
            mcap = info.get('marketCap') or fast_info.get('marketCap')
            if mcap:
                c3.metric("시가총액", f"${mcap / 1_000_000_000:.2f}B")
            else:
                c3.metric("시가총액", "N/A")
            
            # PER
            per = info.get('trailingPE')
            if per:
                c4.metric("PER", f"{per:.2f}")
            else:
                c4.metric("PER", "N/A")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 🌟 2. 이동평균선 계산 (20일, 60일)
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            
            # 🌟 3. 깔끔한 단일 캔들 차트 (거래량 제거)
            fig = go.Figure()
            
            # 캔들차트 추가
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="주가"
            ))
            
            # 이동평균선 추가
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1.5), name="20일 이평선"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='blue', width=1.5), name="60일 이평선"))
            
            fig.update_layout(
                title=f"최근 1년 주가 흐름 및 이동평균선",
                xaxis_rangeslider_visible=False, 
                height=550, 
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis_title="주가 (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"🚨 오류가 발생했습니다: {e}")
