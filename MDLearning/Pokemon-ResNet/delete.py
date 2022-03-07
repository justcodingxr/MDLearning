# -*-coding:utf-8-*-
#检查需要删除的图片,报错位置的上一个
import os
from PIL import Image
folder_path = 'Pokemon'
extensions = []
index=0

for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        print(file_path)
        im = Image.open(file_path)

        rgb_im = im.convert('RGB')
