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
            
            # --- 🛡️ 정보 차단 우회 ---
            info = {}
            fast_info = {}
            try: info = stock.info
            except: pass
            try: fast_info = stock.fast_info
            except: pass

            # 🌟 1. 상단 핵심 요약 (PER 제거, 3열 배치)
            name = info.get('longName', ticker)
            st.subheader(f"{name} ({ticker})")
            
            c1, c2, c3 = st.columns(3)
            
            current_price = hist['Close'].iloc[-1]
            c1.metric("현재가", f"${current_price:.2f}")
            
            high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
            c2.metric("52주 최고가", f"${high_52w:.2f}")
            
            mcap = info.get('marketCap') or fast_info.get('marketCap')
            c3.metric("시가총액", f"${mcap / 1_000_000_000:.2f}B" if mcap else "N/A")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 🌟 2. 차트 영역
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="주가"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='orange', width=1.5), name="20일 이평선"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='blue', width=1.5), name="60일 이평선"))
            
            fig.update_layout(title=f"최근 1년 주가 흐름 및 이동평균선", xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="주가 (USD)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 🌟 3. 펀더멘털 및 재무 실적 요약 (신규 추가)
            st.markdown("### 📊 주요 재무 실적 (연간)")
            try:
                financials = stock.financials
                if not financials.empty:
                    # 필요한 항목 추출
                    metrics = {
                        "Total Revenue": "총 매출액 (Total Revenue)",
                        "Gross Profit": "매출 총이익 (Gross Profit)",
                        "Operating Income": "영업 이익 (Operating Income)",
                        "Net Income": "순이익 (Net Income)"
                    }
                    
                    fin_data = {}
                    for eng, kor in metrics.items():
                        if eng in financials.index:
                            # 보기 쉽게 10억 달러(Billion) 단위로 변환
                            fin_data[kor] = financials.loc[eng] / 1_000_000_000
                    
                    if fin_data:
                        fin_df = pd.DataFrame(fin_data).T
                        # 컬럼(날짜)을 연도만 보이게 깔끔하게 정리
                        fin_df.columns = [str(col).split('-')[0] + "년" for col in fin_df.columns]
                        
                        st.write("*(단위: 10억 달러, Billion USD)*")
                        # 숫자에 쉼표 넣어서 출력
                        st.dataframe(fin_df.style.format("{:,.1f}"), use_container_width=True)
                    else:
                        st.info("해당 기업의 상세 재무 데이터를 불러올 수 없습니다.")
                else:
                    st.info("야후 파이낸스에서 재무 데이터를 제공하지 않습니다.")
            except Exception as e:
                st.info("재무 데이터를 불러오는 데 실패했습니다.")
                
        except Exception as e:
            st.error(f"🚨 오류가 발생했습니다: {e}")
