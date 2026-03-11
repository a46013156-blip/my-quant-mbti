import streamlit as st
import yfinance as yf
import plotly.express as px

def run():
    st.title("📈 개별종목 딥다이브 분석")
    st.markdown("---")
    
    ticker = st.text_input("분석할 종목 티커를 입력하세요 (예: TSLA, AAPL, NVDA)", "").upper()
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("현재가", f"${info.get('currentPrice', 'N/A')}")
                st.write(f"**기업명:** {info.get('longName')}")
            with col2:
                st.write(f"**섹터:** {info.get('sector')}")
                st.write(f"**시가총액:** ${info.get('marketCap', 0):,}")

            df = stock.history(period="1y")
            st.plotly_chart(px.line(df, y="Close", title=f"{ticker} 최근 1년 주가 흐름"), use_container_width=True)
            
        except Exception as e:
            st.error(f"데이터를 불러올 수 없습니다: {ticker}. 티커가 정확한지 확인해주세요.")
