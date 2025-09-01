from PIL import Image, ImageDraw, ImageFont
import os,sys
import glob
import random
import cv2
import numpy as np
import shutil
from tqdm import tqdm
def gen_yolo_format_data(rootpath, dstpath):
    if not os.path.exists(dstpath):
        os.makedirs(dstpath, exist_ok=True)
    list_images = glob.glob(f'{rootpath}/**/*.jpg', recursive=True)
    for imgpath in tqdm(list_images):
        if "/ccpd_np/" in imgpath:#cpd_np是没有车牌的图片，跳过
            continue
        #print(imgpath)
        img = cv2.imread(imgpath)
        imgname = os.path.basename(imgpath).split('.')[0]
        parts = imgname.split('-')
        
        # 先检查分割结果长度
        if len(parts) != 7:  # 确保有7个字段
            print(f"跳过无效文件: {imgpath} - 分割段数={len(parts)}")
            continue
            
        # 安全解包
        _, _, box, points, label, brightness, blurriness = parts

        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]
        box_w = box[1][0]-box[0][0]
        box_h = box[1][1]-box[0][1]
        box = [box[0][0]+box_w/2, box[0][1]+box_h/2, box_w, box_h]
        box = [box[0]/img.shape[1], box[1]/img.shape[0], box[2]/img.shape[1], box[3]/img.shape[0]]
        # --- 关键点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:]+points[:2]
        points = [[pt[0]/img.shape[1], pt[1]/img.shape[0]] for pt in points]
        #print(box, points)

        random_number = random.uniform(0, 1)
        if random_number > 0.2:#train
            dstimgsavefold = os.path.join(dstpath, 'images','train')
            dstlabelsavefold = os.path.join(dstpath, 'labels','train')
        else:
            dstimgsavefold = os.path.join(dstpath, 'images','val')
            dstlabelsavefold = os.path.join(dstpath, 'labels','val')
        os.makedirs(dstimgsavefold, exist_ok=True)
        os.makedirs(dstlabelsavefold, exist_ok=True)
        # --- 保存图像
        cv2.imwrite(os.path.join(dstimgsavefold, imgname+'.jpg'), img)
        # --- 保存标签
        with open(os.path.join(dstlabelsavefold, imgname+'.txt'), 'w') as f:
            f.write(f"{0} {box[0]} {box[1]} {box[2]} {box[3]}")
            for pt in points:
                f.write(f" {pt[0]} {pt[1]}")
            f.write('\n')

        #show_yolo_format_data(img, box, points)
        #break

def show_yolo_format_data(img, bbox, points):
    img_h, img_w, _ = img.shape
    x,y,w,h = bbox
    x1,y1,x2,y2 = (x-w/2)*img_w, (y-h/2)*img_h, (x+w/2)*img_w, (y+h/2)*img_h
    #print(x1,y1,x2,y2)
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    #print(x1,y1,x2,y2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    for pt in points:
        cv2.circle(img, (int(pt[0]*img_w), int(pt[1]*img_h)), 5, (0,0,255), -1)
    cv2.imwrite('img_yolo_format_show.jpg', img)

if __name__ == '__main__':
    if len(sys.argv)!= 3:
        print("Usage: python gen_yolo_format_data.py <ccpd_dataset_path> <output_path>")
        exit(1)
    gen_yolo_format_data(sys.argv[1], sys.argv[2])
