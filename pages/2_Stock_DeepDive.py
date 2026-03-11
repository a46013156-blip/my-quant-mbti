import streamlit as st

st.set_page_config(page_title="Stock Deep Dive", layout="wide")

st.title("📈 개별종목 딥다이브 분석")
st.markdown("---")

st.write("이제 이곳은 ETF와 완전히 분리된 안전 지대입니다! 마음껏 뜯어고쳐도 됩니다.")

ticker = st.text_input("분석할 종목 티커 (예: AAPL, NVDA)", "")
if st.button("데이터 불러오기"):
    if ticker:
        st.success(f"'{ticker}' 종목 분석을 시작할 준비가 되었습니다!")
    else:
        st.warning("티커를 입력해주세요.")
