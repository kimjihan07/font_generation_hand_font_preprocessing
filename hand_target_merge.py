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
    img = Image.fromarray(img).convert('L')
    img = np.array(img)
    img = add_padding(img, image_size=128, pad_value= 255)
    img = Image.fromarray(img).convert('L')

    return img

def add_padding(img, image_size=128, pad_value=None):
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

def rgbtogray(img_path):
    img_path
    img= cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#     print(img.shape)
#     img_gray = 255 - img[:, :, 3]
    img_gray = img
    img_gray.shape

    hand_wr = Image.fromarray(img_gray).convert('L')
    img = hand_wr.resize((128,128), Image.LANCZOS)
    return img



def enhanc(img):
    enhancer = ImageEnhance.Contrast(img)
    cropped_image = enhancer.enhance(1)
    cropped_image = np.array(cropped_image)
    centered_image = Image.fromarray(cropped_image.astype(np.uint8))
    centered_image = bilateralFilter(cropped_image, 15, 15, 30)
    centered_image = Image.fromarray(centered_image).convert('L')
    return centered_image

def target_hand_merge(person_name):
    rows = 28
    cols = 15
    pdf_page=0
    a=0
    c=0
    e=0
    d=0
    n=0
    t=1
    m=0
    pages = [0,1]
    idx = '/home/piai/Ai/project/makeme/random210_idx.txt' # 파일 경로 변경시 경로 변경 필수 !!
    idx = open(idx).read().splitlines()
    # 경로 수정하시길 바랍니다.
    if not os.path.exists("/home/piai/Ai/project/makeme/{}_fonts_merge_FN".format(person_name)):# 해당 경로에 지정 폴더가 없으면 폴더 생성한다.
        os.makedirs("/home/piai/Ai/project/makeme/{}_fonts_merge_FN".format(person_name))
    for page in pages:
        if page == 1:

            for j in range(0, rows):
                if e == 3:
                    break

                else:
                    d = 2*j

                    e = 2*j + 1
                    for i in range(0, cols):
    #                     print("page: {}, d:{}, e: {}, i: {}, n: {}, t: {} ".format(page,d,e,i,n,t))

                        hand_wr = "/home/piai/Ai/project/makeme/{}_fonts/{} ({}).png".format(person_name, person_name,t)
                        hand_wr = rgbtogray(hand_wr)
                        hand_wr = np.array(hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = crop_fontline4(hand_wr, 85)
                        hand_wr = enhanc(hand_wr)
                        hand_wr = np.array(hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = Image.fromarray(hand_wr.astype(np.uint8))
                        # merge전 128*128로 리사이즈된 각 손글씨 이미지가 필요할 시 활성화 해서 사용하길 바람.
                        # 본인이 각 손글씨를 저장하고자하는 경로로 수정해서 사용하길 바람 
#                         hand_wr.save('/home/piai/Ai/project/makeme/KH_fonts_resize/kh ({}).png'.format(t))


                        target = Image.open("/home/piai/Ai/project/makeme/Target_fonts/타겟_{}_{}_{}_cropped_img.png".format(page,d,i))

                        target = np.array(target)
                        target = Image.fromarray(target).convert('L')
                        hand_wr_size =hand_wr.size
                        target_size =target.size
                        merge_img = Image.new('L', (2*hand_wr_size[0],hand_wr_size[1]))
                        merge_img.paste(hand_wr, (0,0))
                        merge_img.paste(target,(hand_wr_size[0],0))
                        merge_img.save("/home/piai/Ai/project/makeme/{}_fonts_merge_FN/1_{}.png".format(person_name, idx[n]),"png")

                        n+=1
                        t +=1


        else:
            for j in range(0, rows):
                if a ==26:
                    break
                else:    
                    a = 2*j

                    c = 2*j + 1

                    for i in range(0, cols):

                        hand_wr = "/home/piai/Ai/project/makeme/{}_fonts/{} ({}).png".format(person_name, person_name,t)
                        hand_wr = rgbtogray(hand_wr)          
                        hand_wr = np.array(hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = crop_fontline4(hand_wr, 85)
                        hand_wr = enhanc(hand_wr)
                        hand_wr = np.array(hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = np.where(np.sum(hand_wr , axis=0) == 0 , 255 , hand_wr)
                        hand_wr = Image.fromarray(hand_wr.astype(np.uint8))
                        # merge전 128*128로 리사이즈된 각 손글씨 이미지가 필요할 시 활성화 해서 사용하길 바람.
                        # 본인이 각 손글씨를 저장하고자하는 경로로 수정해서 사용하길 바람 
#                         hand_wr.save('/home/piai/Ai/project/makeme/KH_fonts_resize/kh ({}).png'.format(t))


                        target = Image.open("/home/piai/Ai/project/makeme/Target_fonts/타겟_{}_{}_{}_cropped_img.png".format(page,a,i))
                        target = np.array(target)
                        target = Image.fromarray(target).convert('L')
                        hand_wr_size =hand_wr.size
                        target_size =target.size
                        merge_img = Image.new('L', (2*hand_wr_size[0],hand_wr_size[1]))
                        merge_img.paste(hand_wr, (0,0))
                        merge_img.paste(target,(hand_wr_size[0],0))

                        merge_img.save("/home/piai/Ai/project/makeme/{}_fonts_merge_FN/1_{}.png".format(person_name, idx[n]),"png")
                        n+= 1
                        t +=1

    print("이미지 생성완료")
