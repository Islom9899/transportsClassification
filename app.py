import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

# Windows 시스템에서 PosixPath 문제 해결
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# 제목 설정
st.title('교통수단 분류 모델')

# 이미지 업로드
file = st.file_uploader('이미지 업로드', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    # 업로드한 이미지 화면에 표시
    st.image(file)

    # 이미지를 PIL 형식으로 변환
    img = PILImage.create(file)

    # 모델 불러오기
    model = load_learner('transport_model.pkl')

    # 예측 수행
    pred, pred_id, probs = model.predict(img)
    st.success(f"예측 결과: {pred}")
    st.info(f"확률: {probs[pred_id]*100:.1f}%")

    # 확률 바 그래프 생성
    fig = px.bar(
        x=probs * 100,
        y=model.dls.vocab,
        orientation='h',
        labels={'x': '확률 (%)', 'y': '클래스'}
    )
    st.plotly_chart(fig)