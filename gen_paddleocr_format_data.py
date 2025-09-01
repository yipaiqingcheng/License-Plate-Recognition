import os, sys
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

provincelist = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", 
                "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "西", "陕", "甘", "青", "宁", "新"]

wordlist = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", 
            "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def gen_paddlecor_format_data(rootpath, dstpath):
    os.makedirs(dstpath, exist_ok=True)
    os.makedirs(os.path.join(dstpath, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dstpath, 'val'), exist_ok=True)
    
    list_images = glob.glob(f'{rootpath}/**/*.jpg', recursive=True)
    
    for imgpath in tqdm(list_images):
        if "/ccpd_np/" in imgpath:  # 跳过无车牌图像
            continue
            
        imgname = os.path.basename(imgpath).split('.')[0]
        parts = imgname.split('-')
        
        # 先检查分割结果长度
        if len(parts) != 7:  # 确保有7个字段
            print(f"跳过无效文件: {imgpath} - 分割段数={len(parts)}")
            continue
            
        # 安全解包
        _, _, box, points, label, brightness, blurriness = parts
        
        # 边界框处理
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]
        box_w = box[1][0] - box[0][0]
        box_h = box[1][1] - box[0][1]

        filename = label
        
        # 车牌号解析
        label_parts = label.split('_')
        province = provincelist[int(label_parts[0])]
        words = [wordlist[int(i)] for i in label_parts[1:]]
        plate_text = province + ''.join(words)
        
        # 裁剪车牌区域
        img = cv2.imread(imgpath)
        img_plate = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        
        # 分割训练/验证集
        random_number = random.uniform(0, 1)
        if random_number > 0.1:  # train
            dst_img_path = os.path.join(dstpath, 'train', f"{filename}.jpg")
            label_file = os.path.join(dstpath, 'train.txt')
            label_content = f"train/{filename}.jpg\t{plate_text}\n"
        else:  # val
            dst_img_path = os.path.join(dstpath, 'val', f"{filename}.jpg")
            label_file = os.path.join(dstpath, 'val.txt')
            label_content = f"val/{filename}.jpg\t{plate_text}\n"
            
        cv2.imwrite(dst_img_path, img_plate)
        with open(label_file, 'a', encoding='utf-8') as f:
            f.write(label_content)

def gent_license_dict(dict_save_path):
    allwordlist = wordlist + provincelist
    allwordlist.sort()
    with open(dict_save_path, 'w', encoding='utf-8') as f:
        for word in allwordlist:
            f.write(word + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python gen_paddlecor_format_data.py <ccpd_dataset_path> <output_path>")
        exit(1)
    
    gen_paddlecor_format_data(sys.argv[1], sys.argv[2])
    gent_license_dict(os.path.join(sys.argv[2], "dict.txt"))