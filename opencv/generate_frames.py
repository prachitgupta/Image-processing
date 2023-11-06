import os
import numpy as np
import cv2
from glob import glob

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error????")

def save_frame(video_path,save_dir,gap = 10):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir,name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    i =0

    while True:
        ret,frame = cap.read()
        if ret == False:
            cap.release()
            break

        if i ==0:
            cv2.imwrite(f"{save_path}/{i+3}.jpg",frame)
        else:
            if i%gap == 0:
                cv2.imwrite(f"{save_path}/{i+3}.jpg",frame)
        i += 1

if __name__ == "__main__":
    video_paths = glob("CARS.mp4")
    save_dir = "datast"

    for path in video_paths:
        save_frame(path,save_dir,gap =12)



    