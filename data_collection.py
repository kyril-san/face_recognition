import numpy as np
import os
import timeit
import tensorflow as tf
import cv2
import uuid

def extract_image_from_file(filePath, NewPath):
    for path in os.listdir(filePath):   
        expath = os.path.join(filePath, path)
        newPath = os.path.join(NewPath, path)
        if expath.endswith((".jpg", ".png", ".jpeg")):
            os.replace(expath, newPath)

def extract_anchor_and_pos_image(filepath, NewPath):
    for file in os.listdir(filepath):
        expath = os.path.join(filepath, file)
        newpath = os.path.join(NewPath, file)
        if file.startswith('1--') and file.lower().endswith((".jpg", "jpeg", ".png")):
            print(file)
            os.replace(expath, newpath)

def augment_images(filepath):
    for file in os.listdir(filepath):
        img_path = os.path.join(filepath, file)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img)
        
        for image in augmented_images:
            cv2.imwrite(os.path.join(filepath, '{}.jpg'.format(uuid.uuid1())), image.numpy())

def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data
           
if __name__ ==  '__main__':
    test_file_path = os.path.join('dataset', 'a.i. faces.v2i.multiclass', 'test')
    train_file_path = os.path.join('dataset', 'a.i. faces.v2i.multiclass', 'train')
    valid_file_path = os.path.join('dataset', 'a.i. faces.v2i.multiclass', 'valid')
    Neg_Path =  os.path.join('data', 'Neg_img')
    img_path = os.path.join('dataset', 'doan3.v10i.folder', 'train', 'quen')
    Anchor_path =  os.path.join('data', 'Anchor_img')
    Pos_path =  os.path.join('data', 'Pos_img')
    extract_image_from_file(test_file_path, Neg_Path)
    extract_image_from_file(train_file_path, Neg_Path)
    extract_image_from_file(valid_file_path, Neg_Path)
    extract_anchor_and_pos_image(img_path, Anchor_path)
    extract_anchor_and_pos_image(img_path, Pos_path)
    augment_images(Anchor_path)
    augment_images(Pos_path)