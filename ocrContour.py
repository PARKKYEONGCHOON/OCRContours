import cv2
import numpy as np
import matplotlib.pyplot as plt

width, height = 640, 480

# 이미지 로드
image = cv2.imread('./img/question.jpg')

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("dsd",gray)

cv2.imshow('image', image)
# 이미지의 노이즈를 제거하기 위해 가우시안 블러를 적용합니다
blurred = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imshow('blurred', blurred)


# Canny 엣지 검출기를 사용하여 엣지 검출
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('edges', edges)

# 이진화 (Threshold) 적용
_, binary = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thre", binary)



# 컨투어 검출
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 원본 이미지에 사각형 그리기
rect_img = image.copy()

for contour in contours:
    
    # 컨투어를 근사화합니다
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
     # 근사화된 컨투어가 4개의 점을 가지고 있을 경우 사각형으로 판단
    if len(approximation) == 4:

        # 각 컨투어의 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(contour)
        # 사각형 그리기
        cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
         # 경계를 사용하여 이미지를 크롭합니다
        cropped_image = image[y:y+h, x:x+w]
        # 결과 이미지를 표시
        #cv2.imshow(f'Cropped Image {i}', cropped_image)
        
        # 투시 변환을 위한 좌표 정렬 (원본 좌표)
        src = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        src = src[np.argsort(src[:, 1])]  # y 좌표에 따라 정렬
            
        # 좌상단, 우상단, 좌하단, 우하단 순서로 좌표 재정렬
        if src[0][0] > src[1][0]:
            src[0], src[1] = src[1], src[0]
        if src[2][0] > src[3][0]:
                src[2], src[3] = src[3], src[2]
            
        # 목표 이미지의 좌표 설정 (목표 좌표)
        dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            
        # 투시 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(src, dst)
        
        cv2.imshow('Detected Rectangles{i}', rect_img)
        
        # 투시 변환 적용
        result = cv2.warpPerspective(image, matrix, (width, height))
        cv2.imshow('result', result)
        
        
        cv2.waitKey()
