# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import os
import glob
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from PIL import ImageFilter
from PIL import ImageEnhance
from cv2 import bilateralFilter
from pdf2image import convert_from_path
from matplotlib import pyplot as plt

def pdf_cropline(img, resize_fix): # pdf를 이미지로 변환 후 
    
    img = np.array(img)
    
    img = img[53:2255, 55:1592]
#     print(img.shape)
    origin_h, origin_w = img.shape
    if origin_h > origin_w:
        resize_w = int(resize_fix * (origin_w/origin_h))
        resize_h = resize_fix
    elif origin_h == origin_w:
        resize_w = resize_fix
        resize_h = resize_fix
    else:
        resize_h = int(resize_fix * (origin_h/origin_w))
        resize_w = resize_fix
    

    img = cv2.resize(img, (resize_w, resize_h), Image.LANCZOS)
    img = Image.fromarray(img).convert('L')

    return img





def crop_fontline4(img, resize_fix):
    
    img = np.array(img)
    img_size = img.shape[0]
    
    full_white = img_size
    col_sum = np.where(np.sum(img, axis=0) < 255 * 128)
    row_sum = np.where(np.sum(img, axis=1) < 255 * 128)
    
    
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    
    img = img[y1:y2, x1:x2]
    origin_h, origin_w = img.shape
    if origin_h > origin_w:
        resize_w = int(resize_fix * (origin_w/origin_h))
        resize_h = resize_fix
    elif origin_h == origin_w:
        resize_w = resize_fix
        resize_h = resize_fix
    else:
        resize_h = int(resize_fix * (origin_h/origin_w))
        resize_w = resize_fix
    

    img = cv2.resize(img, (resize_w, resize_h), Image.LANCZOS)
    
    #     print(type(img))
    img = Image.fromarray(img).convert('L')
    #     print(type(img))
    # plt.imshow(img)
    img = np.array(img)
    # print(img.size)
    # plt.imshow(img)
    img = add_padding(img, image_size=128, pad_value= 255)
    img = Image.fromarray(img).convert('L')
#     plt.imshow(img)
    return img

def crop_fontline1(img):
    
    img = np.array(img)
    img_size = img.shape[0]

    full_white = img_size
    col_sum = np.where(np.sum(img, axis=0) < 255 * 128)
    row_sum = np.where(np.sum(img, axis=1) < 255 * 128)
    
    
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
#     print(y1,y2)
#     print(x1,x2)
    img = img[y1:y2, x1:x2]
    img = add_padding(img, image_size=128, pad_value= 255)
    img = Image.fromarray(img).convert('L')
    return img


def add_padding(img, image_size=128, pad_value=None): # crop_fontlin1에 포함되어있음. padding함수 정의 필수 
    #인풋 넘파이 
    height, width = img.shape
    if not pad_value:
        pad_value = 255
    
    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)
    
    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)
    
    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)
    
    return img

def rgbtogray(img_path): # 읽어올 이미지가 gray scale이 아닐 경우 ex RGBA,RGB로 변경하는 함수
    img_path
    img= cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#     print(img.shape)
#     img_gray = 255 - img[:, :, 3]
    img_gray = img
    img_gray.shape

    hand_wr = Image.fromarray(img_gray).convert('L')
    img = hand_wr.resize((128,128), Image.LANCZOS)
    return img


def enhanc(img): # 이미지 대비를 변경하여 글씨의 진하기를 높여주는 함수
    enhancer = ImageEnhance.Contrast(img)
    cropped_image = enhancer.enhance(1)
    cropped_image = np.array(cropped_image)
    centered_image = Image.fromarray(cropped_image.astype(np.uint8))
    centered_image = bilateralFilter(cropped_image, 15, 15, 30)
    centered_image = Image.fromarray(centered_image).convert('L')
    return centered_image



def pdf_crop_handfonts(pdf_path, person_name):  # 템플릿.pdf에서 타겟 폰트를 제외한 손글씨를 png로 크롭하는 함수임.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------    
# 해당 레포지토리를 그대로 git hub에서 clone할 경우 바로 실행될 것임.    
# 아래의 변수들의 경우 위의 경로를 사용하시는 워크스테이션 or pc에 고딕체 등 ttf 파일 저장되어 있는 경로로 꼭 변경하시오.    
    
    pdf = convert_from_path(pdf_path) # pdf파일을 이미지로 변환 convert_form_path 메소드 검색해 볼것
    
    scr='/home/piai/Ai/project/handwriter/font_generation/font/source/NanumGothic.ttf' # 사용할 컴퓨터 폰트 경로 ttf 파일이어야함.
    

    scr_font = ImageFont.truetype(scr, size = 80) # 타겟이미지 생성시 128* 128 사이즈의 흰색 이미지에 쓰여질 타겟이미지 크기 : 80사이즈가 가장 적당함.
    
    charset = '/home/piai/Ai/project/makeme/random210_cha.txt' # 템플릿에 적힌 각 글자가 적힌 텍스트 파일 
    
    charset = open(charset).read().splitlines() # charset파일을 줄단위로 읽어서 리스트로 반환
    
    idx = '/home/piai/Ai/project/makeme/random210_idx.txt' # char의 idx 순서 charset과 동일
    
    idx = open(idx).read().splitlines() # idx파일을 줄단위로 읽어서 리스트로 반환
    
    rows = 28 # 주어진 템플릿의 총 행수가 28개 # 그중 0포함 짝수행은 컴퓨터 폰트, 즉 타겟 폰트이므로 pdf에서 폰트를 직접 crop하지 않음.
    
    cols = 15 # 주어진 템플릿의 총 열수가 15개
    
    m = 1 # 템플릿의 총 글자수가 240개이므로 1~240개를 세기위한 변수임. 
    
    pdf_page=0 # pdf페이지가 총 2장인데, 2페이지의 경우 행이 4개 밖에 되지않으므로 2페이지에서 읽어오는 행을 3행까지로 제한하기위한 변수
    
    
    
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------    
    
    # if문 안에 있는 경로도 꼭 본인의 사용환경에 맞게 변경하시오.
    if not os.path.exists("/home/piai/Ai/project/makeme/{}_fonts".format(person_name)):# 해당 경로에 지정 폴더가 없으면 폴더 생성한다.
        os.makedirs("/home/piai/Ai/project/makeme/{}_fonts".format(person_name))
        
    
    for page in pdf: # p

        page = page.convert('L') # pdf 페이지를 흑백이미지로 변경
        page = pdf_cropline(page, 26424)# 주어진 Template의 윤곽선을 제외한 부분 img 좌표 # pdf_cropline메서드의 경우


        np_page = np.array(page) # 이미지로 변환된 pdf 파일을 array로 저장
        cropped_img = np.where(np_page < 191,0,255) #127보다 작으면 0 ,127보다 크면 255
        pdf = Image.fromarray(cropped_img.astype(np.uint8))
       
        width, height = pdf.size
        cell_width = width/float(cols) # 이것은 pdf의 가장 바깥의 윤곽선을 제외한 페이지의 width에서 열의 갯수를 나눈 값으로 글자칸당 width을 의미함: 열을 나눌때 만 사용
        if pdf_page ==1: # pdf 페이지 2페이지 인 경우 
            for j in range(0,rows): # 행이 총 4개 이므로 0,1,2,3 까지 읽어 온 후 4에서 반복문 종료 
                if j == 4:
                        break

                elif j % 2 == 0: # 짝수 행 타겟 소스 행    이 조건문의 경우 손글씨 crop에 직접적인 영향을 주진 않으나  손글씨 이미지를 crop 하는 조건문에 각각의 손글씨 이미지의 위치를 전달하므 변경하지말것 
                    for i in range(0, cols):
                        left = i * cell_width
                        upper = j * 942 # 템플릿의 형태를 그대로 사용할 경우 절대 변경하지마시오.
                        right = left + cell_width
                        lower = upper + 660 # 템플릿의 형태를 그대로 사용할 경우 절대 변경하지마시오.
                        connect_num = lower

                elif j % 2 != 0:

                    for i in range(0,cols): # pdf에서 손글씨만 crop하는 if문 아래의 저장경로는 본인의 사용환경에 맞게 수정할 것


                        left = i * cell_width
                        upper = connect_num 
                        right = left + cell_width
                        lower = upper + 1224 # 템플릿의 형태를 그대로 사용할 경우 절대 변경하지마시오.
                        cropped_image = pdf.crop((left,upper,right,lower))
                        cropped_image.save("/home/piai/Ai/project/makeme/{}_fonts/{} ({}).png".format(person_name,person_name,m)) # 
                        m +=1

        else: # pdf 1페이지의 경우 

            for j in range(0,rows):
                if j % 2 == 0: # 짝수 행 타겟 소스 행    :: 이 조건문의 경우 손글씨 crop에 직접적인 영향을 주진 않으나  손글씨 이미지를 crop 하는 조건문에 각각의 손글씨 이미지의 위치를 전달하므 변경하지말것 

                    for i in range(0, cols):

                        left = i * cell_width
                        upper = j * 942 #  기존 471
                        right = left + cell_width
                        lower = upper + 660 # 기존 330
                        connect_num = lower

                elif j % 2 != 0: # 홀수 행 핸드라이팅 행   # pdf에서 손글씨만 crop하는 if문 아래의 저장경로는 본인의 사용환경에 맞게 수정할 것 
                    for i in range(0,cols):


                        left = i * cell_width
                        upper = connect_num 
                        right = left + cell_width
                        lower = upper + 1224 # 기존 306
                        cropped_image = pdf.crop((left,upper,right,lower))
                        cropped_image.save("/home/piai/Ai/project/makeme/{}_fonts/{} ({}).png".format(person_name,person_name,m))
                        m += 1                    
        pdf_page += 1
    print('폰트 생성이 완료되었습니다.')