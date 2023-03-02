import streamlit as st
file=st.file_uploader("上传文件")
if __name__ == '__main__':
    if file is not None:
        st.image(file)