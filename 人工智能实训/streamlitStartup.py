import json
import os
import shutil
from io import BytesIO

import streamlit as st
import torch
import torchvision.models.googlenet
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from PIL import Image




def googlenet():
    st.title("GoogleNet模型")
    file = st.file_uploader("上传图片(.jpg)", type=".jpg")
    model = torchvision.models.googlenet(pretrained=True)

    with open('imagenet-labels.json', 'r') as f:

        data = json.load(f)  # labels = [line.strip() for line in f.readlines()]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    model.eval()

    def predict(img):
        img_t = transform(img)
        output = model(img_t.unsqueeze(0))
        _, pred = torch.max(output, 1)
        sorted, indices = torch.sort(output, descending=True)

        return data[pred[0]], sorted.squeeze()[:10], indices.squeeze()[:10]

    if file is not None:
        img=Image.open(file)
        st.image(img)
        label,sortedtensor,indices =predict(img)
        import plotly.graph_objs as go

        print(indices.tolist())
        # 柱状图的标签
        labels =[data[i] for i in indices.tolist()]


        # 柱状图的值
        values = sortedtensor.tolist()

        # 创建柱状图的数据
        data = [go.Bar(x=labels, y=values)]

        # 创建柱状图的布局
        layout = go.Layout(title='识别结果', xaxis=dict(title='类别'), yaxis=dict(title='置信度'))

        # 创建柱状图的Figure
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
        st.write(str(label))

# Define page 2
def yolov5():
    project=os.getcwd()
    st.title("YOLOv5")
    file = st.file_uploader("上传图片或视频")
    if file is not None:
        st.markdown("### 原图")
        st.image(file)

        file_bytes=BytesIO(file.read())
        filepath="yolo_tmp/"+file.name.title()
        outputpath="yolo_tmp/output_"+file.name.title()
        with open(filepath,"wb") as f:
            f.write(file_bytes.getbuffer())
        detect.run(source=filepath,project=project,name=outputpath)
        output_img=Image.open(outputpath+"/"+str(file.name.title()))
        st.markdown("### YOLOv5识别结果：")
        st.image(output_img)
        shutil.rmtree(outputpath)
        os.remove(filepath)
pages = {
    "GoogleNet（支持图片）": googlenet,
    "YOLOv5（支持图片和视频）": yolov5,
}
if __name__ == "__main__":
    import sys

    sys.path.append("yolov5-master")
    import detect

    option = st.sidebar.selectbox('选择一个模型',['GoogleNet（支持图片）','YOLOv5（支持图片和视频）'])
    page=pages[option]
    page()



