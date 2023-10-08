import os
import numpy as np

def file_type(file_name):
    # 获取文件名的后缀部分，即'.mp4'等
    if os.path.isdir(file_name):
        return 'folder'
    ext = os.path.splitext(file_name)[1]
    if ext in [".mp4", ".avi", ".mov"]:
        return 'video'
    elif ext in [".jpg", ".png", ".jpeg",".JPG",".PNG",".JPEG"]:
        return 'image'
    else:
        raise("file type not supported")
    
def create_incremental_folder(base_name="results/video_results"):
    counter = 1
    while True:
        folder_name = f"{base_name}{counter}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        counter += 1
    
    
