import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('교통 수단을 분류하는 모델')

# 사진 입력
file = st.file_uploader('사진 입력',type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    model = load_learner('transport_model.pkl')

    #prediction
    pred,pred_id,probs = model.predict(img)
    st.success(f"예측 결과: {pred}")
    st.info(f"확률: {probs[pred_id]*100:.1f}%")
    
    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
