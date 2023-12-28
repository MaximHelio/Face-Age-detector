import numpy as np
import cv2
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

# Caffe프레임워크에서 사용되는 모델의 가중치 파일
face_model = './model/res10_300x300_ssd_iter_140000.caffemodel'
face_prototxt = './model/deploy.prototxt.txt'
age_model = './model/age_net.caffemodel'
age_prototxt = './model/age_deploy.prototxt'
gender_model = './model/gender_net.caffemodel'
gender_prototxt = './model/gender_deploy.prototxt'
image_file = './image/male1.jpg'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

title_name = 'Age and Gender Recognition'
# 50%이상의 정확도
min_confidence = 0.5
min_likeness = 0.5
OUTPUT_SIZE = (300, 300)

# 3개의 detector 객체 생성
detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)


def selectFile():
    file_name = filedialog.askopenfilename(
        # jpg 파일만 select
        initialdir="./image", title="Select file", filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    read_image = cv2.imread(file_name)
    # bgr을 rgb로 변환
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    # RGB형식의 이미지를 PIL(Python Imaging Library) 객체로 변환
    image = Image.fromarray(image)
    # PIL의 Image 객체를 Tkinter에서 사용할 수 있는 형식인 PhotoImage객체로 변환
    imgtk = ImageTk.PhotoImage(image=image)
    # OpenCV에서 이미지의 높이와 너비를 읽어와서 해당 정보를 변수에 할당하는 코드
    (height, width) = read_image.shape[:2]
    fileLabel['text'] = file_name
    detect(read_image)

# 읽어온 이미지 검증하기


def detect(image):
    # image.shape(높이, 너비, 채널)
    (h, w) = image.shape[:2]
    # 모델이 이미지 처리할 수 있는 형식으로 변환함 scale = 1.0 : 이미지의 크기 비율 지정, size, 윤곽의 color이미지가 필요한 것이므로
    # swapRB: RBG로 바꿀 거냐 뭐냐, crop: 이미지 자를 거니?/여기선 전체 이미지 다 사용
    # 이 그림에서 얼굴만 찾아줌
    imageBlob = cv2. dnn.blobFromImage(
        cv2.resize(image, OUTPUT_SIZE), 1.0, OUTPUT_SIZE, (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    # 새로 그림 넣을 때마다 초기화, 첫번째 행(1)의 0번째 열(0)을 나타냄
    log_ScrolledText.delete(1.0, END)
    # 여러 개의 bounding box(경계상자)는 감지된 무체 또는 얼굴에 대한 정보를 나타냄.
    for i in range (0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            #  - 3번째 detection[0,0,i,3]는 전체 폭 중 박스 시작점의 x좌표 상대위치 (왼쪽 맨 위 시작점)
            # - 4번째 detection[0,0,i,4]는 전체 높이 중 박스 시작점의 y좌표 상대위치
            # - 5번째 detection[0,0,i,5]는 전체 폭 중 박스 끝점의 x좌표 상대위치  (오른쪽 맨 아래 끝점)
            # - 6번째 detection[0,0,i,6]는 전체 높이 중 박스 끝점의 y좌표 상대위치
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            face = image[startY:endY, startX:endX]
            (fH, fW)=face.shape[:2]
            # 얼굴 자르기
            # (입력이미지데이터, scalefactor:이미지 크기 조절 비율, size:결과 블롭크기, RGB평균값, RGB를 RBG로 바꿀 것인지)
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (978.42633603, 87.7689143744, 114.89847746), swapRB=False)
            # 얼굴 1개니까 0번째 인덱스, age중에 probability 가장 높은 것이 뭐냐
            age_detector.setInput(face_blob)
            age_predictions = age_detector.forward()
            age_index = age_predictions[0].argmax()
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]
            # 성별도 나이와 같이
            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            # 포맷에 맞게 출력값 만들어줌
            text = "{}:{}".format(gender, age)
            # 얼굴 윤곽에 10만큼 공간이 없으면 아래다가 써줌, 아니면 위에다 써줌
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            # TITLE: 오렌지색
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % (
            'Gender: ', gender, gender_confidence*100, '%') + '\n\n', 'TITLE')
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % (
                    'Age: ', age, age_confidence*100, '%') + '\n\n', 'TITLE')
            log_ScrolledText.insert(END, "%15s %20s" % (
                    'Age', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(age_list)):
                    log_ScrolledText.insert(END, "%10s %15.2f" % (
                        age_list[i], age_predictions[0][i]*100)+'\n')
            log_ScrolledText.insert(END, "%12s %20s" % (
                'Gender', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(gender_list)):
                log_ScrolledText.insert(END, "%10s %15.2f" % (
                    gender_list[i], gender_predictions[0][i]*100)+'\n')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image = imgtk


# Tkinter 라이브러리 사용해서 GUI 창을 생성
main = Tk()
# GUI창의 제목 설정
main.title(title_name)
# 화면 구성
main.geometry()

# load the input image and convert it from BGR to RGB
read_image = cv2.imread(image_file)
(height, width) = read_image.shape[:2]
# openCV는 bgr format(색맹 검사지 느낌)이므로 -> rgb포맷으로 바꾸어줌
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB, )
# 화면에 보여줌
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)
label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
fileLabel = Label(main, text=image_file)
fileLabel.grid(row=1, column=0, columnspan=2)
Button(main, text="File Select", height=2, command=lambda: selectFile()).grid(
    row=1, column=2, columnspan=2, sticky=(N, S, W, E))
detection = Label(main, image=imgtk)
detection.grid(row=2, column=0, columnspan=4)

log_ScrolledText = tkst.ScrolledText(main, height=20)
log_ScrolledText.grid(row=3, column=0, columnspan=4, sticky=(N, S, W, E))

log_ScrolledText.configure(font='TkFixedFont')

log_ScrolledText.tag_config(
    'HEADER', foreground='gray', font=("Helvetica", 14))
log_ScrolledText.tag_config('TITLE', foreground='orange', font=(
    "Helvetica", 18), underline=1, justify='center')

detect(read_image)
# TKinter의 이벤트 루프를 시작. 사용자가 윈도우를 조작할 때 발생하는 이벤트에 대한 응답을 처리
main.mainloop()
