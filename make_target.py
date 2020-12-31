from __future__ import print_function
from __future__ import absolute_import
# import crop
# from crop import crop_fontline, add_padding
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
#--------------------------------------------------------------------------------------------------------------
# 주의!!!!!!!!!
# 만약 아래의 코드를 바꾸시게 되면, 추후 타겟폰트이미지와 손글씨이미지를 merge하는 과정에서 오류가 발생하오니 이점 숙지하시길 바랍니다.

def make_target(make_font_dir): #모델에 넣을 merge 이미지를 만들기 전에 고딕체로 작성된 target font image 생성    
    pages =[0,1]
    # rows, cols는 처음에 코드를 짤때 template crop 기준으로 짜서 들어가게된 변수입니다. targetfonts를 만드는 이코드에서는 파일명을 정할 때 이용됩니다.
    # 만약이코드를 바꾸시게 되면, 추후 타겟폰트이미지와 손글씨이미지를 merge하는 과정에서 오류가 발생하오니 이점 숙지하시길 바랍니다.
    rows = 28 # 코드를 간단하게 만들면 필요없는 변수 
    cols = 15 # 코드를 간단하게 만들면 필요없는 변수
    
    pdf_page=0 #코드를 간단하게 만들면 필요없는 변수
    scr='/home/piai/Ai/project/handwriter/font_generation/font/source/NanumGothic.ttf'
    scr_font = ImageFont.truetype(scr, size = 80) # scr에 저장된 폰트를 새로운 이미지를 그릴때 사용하겠다.
    # 아래의 offset의 경우 20번 이상의 실험을 통해 얻은 이상치이므로 큰 문제가 발생하지 않는 한 바꾸지 마시오.
    x_offset = 20 # 128 *128의 백지 이미지에 글자를 쓸때 여백을 결정하는 변수임. 가로
    y_offset = 20 # 128 *128의 백지 이미지에 글자를 쓸때 여백을 결정하는 변수임. 세로
    charset = '/home/piai/Ai/project/makeme/random210_cha.txt' # 제일 중요! 해당파일에 저장된 글씨를 이미지에 쓰는 것임.
    charset = open(charset).read().splitlines()
    m =0 # m은 0 ~ 239까지 증가함 : 그이유는 손글씨 템플릿의 총 글자수가 240개이기 때문, idx, charset의 행수와 동일함 
    for page in pages: # pages list에서 
        # 아래의 경로는 타겟 폰트가 저장되는 경로입니다.
        if not os.path.exists("{}".format(make_font_dir)):# 해당 경로에 지정 폴더가 없으면 폴더 생성한다.
            os.makedirs("{}".format(make_font_dir))  # 해당 경로는 본인의 환경에 맞게 꼭 변경하시길 바랍니다.

        if pdf_page ==1:
            for j in range(0,rows):
                if j == 4:
                        break

                elif j % 2 == 0: # 짝수 행 타겟 소스 행

                    for i in range(0, cols):
                        ch = charset[m]
                        target = Image.new("RGB", (128,128), (255, 255, 255)).convert('L')
                        draw = ImageDraw.Draw(target)
                        draw.text((x_offset, y_offset), ch, (0), font=scr_font)
                        target.save("{}/타겟_{}_{}_{}_cropped_img.png".format(make_font_dir,pdf_page,j,i))
                        m +=1

                elif j % 2 != 0: # 홀수 행 핸드라이팅 행              

                    for i in range(0,cols):

                        continue

        else:  
            for j in range(0,rows):
                if j % 2 == 0: # 짝수 행 타겟 소스 행

                    for i in range(0, cols):

                        ch = charset[m]
                        target = Image.new("RGB", (128,128), (255, 255, 255)).convert('L')
                        draw = ImageDraw.Draw(target)
                        draw.text((x_offset, y_offset), ch, (0), font=scr_font)
                        target.save("{}/타겟_{}_{}_{}_cropped_img.png".format(make_font_dir,pdf_page,j,i))
                        m+=1
                        
                        
                elif j % 2 != 0: # 홀수 행 핸드라이팅 행              
                    for i in range(0,cols):
                        continue                
        pdf_page += 1
    print('타겟폰트 생성이 완료되었습니다.')
