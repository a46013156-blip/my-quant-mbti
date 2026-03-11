import streamlit as st

st.set_page_config(page_title="My Quant Portfolio", layout="wide")

st.title("📊 My Quant Portfolio")
st.markdown("---")
st.write("환영합니다! 원하시는 분석 도구를 아래에서 클릭하여 이동해주세요.")

st.markdown("<br>", unsafe_allow_html=True)

# 🌟 수직으로 배열된 '진짜' 클릭 이동 버튼
st.page_link("pages/1_ETF_Recipe.py", label="⚖️ [1번 방] ETF 자산배분 모델로 이동하기", icon="1️⃣")
st.page_link("pages/2_Stock_DeepDive.py", label="📈 [2번 방] 개별종목 딥다이브 분석으로 이동하기", icon="2️⃣")
