import os
import sys
import io
import re
import asyncio


# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import curses

import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predicter import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, \
    get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

def get_user_input(prompt):
    return curses.wrapper(_get_user_input, prompt)
    
def _get_user_input(stdscr, prompt):
    curses.echo()
    stdscr.clear()
    stdscr.addstr(0, 0, prompt)
    stdscr.refresh()
    user_input = stdscr.getstr(1, 0).decode('utf-8')
    curses.noecho()
    return user_input

def prepare():
    if 'face_swapper' not in roop.globals.frame_processors:
        roop.globals.frame_processors.append('face_swapper')
    if 'face_enhancer' not in roop.globals.frame_processors:
        roop.globals.frame_processors.append('face_enhancer')
    print(roop.globals.frame_processors)
    execution_providers = []
    if 'cuda' not in execution_providers:
        execution_providers.append('cuda')
        roop.globals.execution_providers = decode_execution_providers(execution_providers)
        print(roop.globals.execution_providers)

    roop.globals.keep_fps = False
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = True
    roop.globals.video_encoder = 'libx264'
    roop.globals.video_quality = 18
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_threads = suggest_execution_threads()


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(),
                                                                     encode_execution_providers(
                                                                         onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 10


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 4


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        torch.cuda.empty_cache()


# 初始化
prepare()

def list_images_in_directory(directory,img):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if '_' not in file and file not in img:
                image_files.append(file_path)
    return image_files


# 指定要遍历的目录路径
#rrr = get_user_input("请输入图像文件名: ")
rrr = '6'
directory_path = './target/'+rrr
img = './target/'+rrr+'/'+rrr+'.jpg'
# 获取目录下所有图片文件的列表
image_files_list = list_images_in_directory(directory_path,img)
print('图片',img)
print('目标目录',directory_path)
print('目标目录文件',image_files_list)
if os.path.exists(directory_path):
    print("存在:", directory_path)
else:
    print("不存在:", directory_path)
if image_files_list:
    # 如果查询结果有数据，则进行处理
    for image_file in image_files_list:
        print('目标图片1',image_file)
        file = image_file
        
        file_name1, file_extension2 = os.path.splitext(os.path.basename(img))
        
        file_name, file_extension = os.path.splitext(os.path.basename(image_file))
        
        # 构造新的文件名
        new_file_name = f"{file_name}{file_extension}"
        new_file_path = os.path.join(os.path.dirname(image_file), file_name1+'_'+new_file_name)
        print('目标图片1',new_file_path)
        if os.path.exists(new_file_path):
            continue
        result = new_file_path

        isExecSuccess = False

        roop.globals.source_path = img
        roop.globals.target_path = file
        roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, result)
        dir_path = os.path.dirname(roop.globals.output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(roop.globals.source_path)
        print(roop.globals.target_path)
        print(roop.globals.output_path)
        try:
            for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                print('frame_processor',frame_processor)
                if not frame_processor.pre_start() or not frame_processor.pre_check():
                    print('检查失败')
            if has_image_extension(file):
                #roop.globals.frame_processors = []
                shutil.copy2(file, result)
                for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                    frame_processor.process_image(img, result, result)
                    frame_processor.post_process()
                    release_resources()
            else:
                print(1111)
                create_temp(roop.globals.target_path)
                print(2222)
                extract_frames(roop.globals.target_path)
                print(3333)
                temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
                print(444)
                for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                    frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
                    frame_processor.post_process()
                    release_resources()
                # handles fps
                print(555)
                if roop.globals.keep_fps:
                    fps = detect_fps(roop.globals.target_path)
                    create_video(roop.globals.target_path, fps)
                else:
                    create_video(roop.globals.target_path)
                print(66)
                # handle audio
                if roop.globals.keep_audio:
                    restore_audio(roop.globals.target_path, roop.globals.output_path)
                else:
                    move_temp(roop.globals.target_path, roop.globals.output_path)
                # clean and validate
                clean_temp(roop.globals.target_path)
            isExecSuccess = True
        except Exception as e:
            isExecSuccess = False
            print('生成失败:', e)

