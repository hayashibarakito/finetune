from glob import glob
from os.path import join
import random
import cv2
import numpy as np
from PIL import Image
import shutil

for i in range(10):
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    name = classes[i]

    img_src = "./imagenet_c/query/" + name
    # 指定パスのPNG画像ファイルのリストを取得
    files = glob(join(img_src, "*.jpg"))
    # ファイルの総数を取得
    num_files = len(files)
    # ファイルのリストを5:5に分ける
    files1 = random.sample(files, int(num_files*0.2))

    #print(files1)
    #print(len(files1))

    for path1 in files1:
        shutil.copy(path1, './imagenet_c/target_domain(20)/' + name )

