import streamlit as st, os, shutil
from PIL import Image

proc, review = "data/processed", "data/review"
os.makedirs(review, exist_ok=True)
imgs = sorted([f for f in os.listdir(proc) if f.endswith(".jpg")])

i = st.slider("Index", 0, len(imgs) - 1, 0)
path = os.path.join(proc, imgs[i])
st.image(Image.open(path), caption=imgs[i])

col1, col2 = st.columns(2)
with col1:
    if st.button("Delete"):
        os.remove(path)
with col2:
    if st.button("Move to review"):
        shutil.move(path, os.path.join(review, imgs[i]))
