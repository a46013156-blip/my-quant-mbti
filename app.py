import streamlit as st

# 분리해둔 2개의 파이썬 파일을 도구구함처럼 불러옵니다.
import etf_app
import stock_app

st.set_page_config(page_title="My Quant Portfolio", layout="wide")

# 사이드바 메뉴 강제 고정 (오류 원천 차단)
st.sidebar.title("📊 My Quant Portfolio")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "원하시는 분석 도구를 선택하세요:", 
    ["⚖️ ETF 자산배분 모델", "📈 개별종목 딥다이브"]
)

# 메뉴 선택에 따라 각 파일의 run() 함수를 실행합니다.
if menu == "⚖️ ETF 자산배분 모델":
    etf_app.run()
elif menu == "📈 개별종목 딥다이브":
    stock_app.run()
