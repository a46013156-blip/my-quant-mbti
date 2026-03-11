import streamlit as st

st.set_page_config(page_title="My Quant Portfolio", layout="wide")

st.title("📊 My Quant Portfolio")
st.markdown("---")
st.write("환영합니다! 이 곳은 데이터 기반의 퀀트 투자 레시피를 설계하는 공간입니다.")
st.info("👈 화면 왼쪽의 사이드바에서 원하시는 분석 도구를 선택해주세요.\n\n"
        "- **1_ETF_Recipe**: 10년 이상 검증된 데이터 기반 자산배분 모델\n"
        "- **2_Stock_DeepDive**: 개별종목 딥다이브 분석 (개발 중)")
