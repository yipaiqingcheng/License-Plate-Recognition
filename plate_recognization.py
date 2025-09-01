import numpy as np
import cv2
import csv
import os
from ocr_rec import TextRecognizer, init_args
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

class PlateRecognizer:
    def __init__(self, det_model_path, rec_args):
        self.model_det = YOLO(det_model_path)
        self.model_ocr = TextRecognizer(rec_args)

    def recognize(self, img):
        plate_objs = []
        try:
            plates = self.model_det(img, verbose=False)
            if len(plates) == 0 or len(plates[0].boxes) == 0:
                return plate_objs

            for box, conf in zip(plates[0].boxes.xyxy, plates[0].boxes.conf):
                x1, y1, x2, y2 = map(int, box.cpu())
                plate_img = img[y1:y2, x1:x2]

                # 车牌识别
                try:
                    rec_res, _ = self.model_ocr([plate_img])
                    text = rec_res[0][0] if len(rec_res) > 0 and len(rec_res[0]) > 0 else ""
                    score_text = rec_res[0][1] if len(rec_res) > 0 else 0.0
                except Exception as e:
                    print(f"OCR error: {e}")
                    text, score_text = "", 0.0

                obj = {
                    'text': text,
                    'score_text': score_text,
                    'bbox': [x1, y1, x2, y2],
                    'score_bbox': conf.cpu().numpy().item()
                }
                plate_objs.append(obj)
        except Exception as e:
            print(f"Detection error: {e}")
        return plate_objs


def DrawPlateNum(img, plate_text, x1, y1, font_size=40):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simsun.ttc", font_size)
    except IOError:
        print("Can't find simsun.ttc, use default font")
        font = ImageFont.load_default()
    draw.text((x1, y1 - font_size), plate_text, font=font, fill=(255, 255, 0))
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr


def process_video(video_path, det_model_path, output_path, rec_args, target_size=None):
    """
    :param video_path: input path
    :param det_model_path: model path
    :param output_path: output path
    :param rec_args: OCR model parameters
    :param target_size: (width hight)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can't open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_size:
        out_width, out_height = target_size
    else:
        out_width, out_height = width, height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # saved to csv file
    csv_path = output_path.rsplit('.', 1)[0] + '.csv'
    csv_file = open(csv_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['frame_id', 'text', 'score_text', 'score_bbox', 'x1', 'y1', 'x2', 'y2'])

    plate_rec = PlateRecognizer(det_model_path, rec_args)

    frame_count = 0
    print(f"Start to process video, total_frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Optional: scaling frames to speed up processing
        if target_size:
            frame = cv2.resize(frame, target_size)

        plate_objs = plate_rec.recognize(frame)

        for obj in plate_objs:
            x1, y1, x2, y2 = obj['bbox']
            text = obj['text']
            score_text = obj['score_text']
            score_bbox = obj['score_bbox']

            csv_writer.writerow([frame_count, text, score_text, score_bbox, x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = DrawPlateNum(frame, text, x1, y1)

        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Process {frame_count}/{total_frames} frames")


    csv_file.close()
    cap.release()
    out.release()
    print(f"Video has been saved to: {output_path}")
    print(f"Recognition results have been saved to: {csv_path}")


if __name__ == "__main__":
    VIDEO_PATH = r"D:\ni\bo\recognization\video_data\32.31.250.103\20240501_20240501140806_20240501152004_140807.mp4"
    DET_MODEL_PATH = r"D:\ni\bo\recognization\license_models\y11n-pose_plate_best.onnx"
    OUTPUT_PATH = r"D:\ni\bo\recognization\video_data\32.31.250.105\result_video_140807.mp4"
    OCR_MODEL_PATH = r"D:\ni\bo\recognization\license_models\license_ocr.onnx"
    CHAR_DICT_PATH = r"D:\ni\bo\recognization\license_models\dict.txt"
    # ==============================

    parser = init_args()


    args = parser.parse_args(args=[]) 


    args.rec_model_dir = OCR_MODEL_PATH
    args.rec_char_dict_path = CHAR_DICT_PATH
    args.use_gpu = True 
    args.gpu_id = 0
    args.rec_batch_num = 4
    args.rec_image_shape = "3, 48, 320"



    target_resolution = None  # for example (1280, 720)


    process_video(
        video_path=VIDEO_PATH,
        det_model_path=DET_MODEL_PATH,
        output_path=OUTPUT_PATH,
        rec_args=args,
        target_size=target_resolution
    )