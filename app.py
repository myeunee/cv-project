import cv2
import streamlit as st
from PIL import Image
import numpy as np

import glob
import wget
import torch
import os
import time

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25

# streamlit UI
def main() :
    st.title('도로침수심 분석 모델')
    st.sidebar.title("Settings")

    # Open the video file
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define the region of interest (ROI)
    roi_x1 = 64
    roi_y1 = 273
    roi_x2 = 297
    roi_y2 = 690



    # 비디오를 캡처해서 이미지를 한 장씩 가져오는 loop
    while cap.isOpened():
        # 1장의 이미지에 대해 객체 탐지
        success, frame = cap.read()

        if success:
            # Crop the ROI from the frame
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # Run YOLOv8 inference on the ROI
            results = model(roi_frame)

            # Visualize the results on the ROI
            annotated_roi_frame = results[0].plot()

            # Merge the annotated ROI frame with the original frame
            annotated_frame = frame.copy()
            annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi_frame

        
            # 현재 frame을 윈도우에 출력
            img = cv2.imshow("YOLOv8 Inference", annotated_frame)

            #파일을 opencv이미지로 변환
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            st_image = cv2.imdecode(file_bytes, 1)

            #이미지를 streamlit에 표시
            st.image(st_image, channels="BGR", caption="객체탐지")
            

            # Break the loop if 'q' is pressed
            # else: Break the loop if the end of the video is reached
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:  
                break
        
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass