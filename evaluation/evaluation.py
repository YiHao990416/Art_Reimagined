import cv2
import numpy as np

import scipy
from scipy.spatial.distance import mahalanobis

line='='*100

def gan_eval(gray_img_dataset, gray_img_generate):
    #convert to histogram
    hist_dataset = []
    hist_generate = []

    for img in gray_img_dataset:
        hist = cv2.calcHist([img], mask=None, channels=[0], histSize=[256], ranges=[0,256])
        hist_dataset.append(hist)
    for img in gray_img_generate:
        hist = cv2.calcHist([img], mask=None, channels=[0], histSize=[256], ranges=[0,256])
        hist_generate.append(hist)

    hist_dataset = np.array(hist_dataset)
    hist_generate = np.array(hist_generate)

    bhattacharyya_score = calculate_BD(hist_dataset,hist_generate)
    d1_score = calculate_d1(hist_dataset,hist_generate)
    d2_score = calculate_d2(hist_dataset,hist_generate)
    d4_score = calculate_d4(hist_dataset,hist_generate)
    d6_score = calculate_d6(hist_dataset,hist_generate)

    # Summary
    print(f'{line}\nResult:')
    print(f'Bhattacharyya distance : {bhattacharyya_score:.8f}')
    print(f'D1 distance : {np.average(d1_score):.8f}')
    print(f'D2 distance : {np.average(d2_score):.8f}')
    print(f'D4 distance : {np.average(d4_score):.8f}')
    print(f'D6 distance : {d6_score:.8f}')
    print(line)

def cumulative_histogram(hist):
    if len(hist.shape)<=1:
        return hist
    cumulated = np.zeros(256)
    for x in hist:
        val = np.squeeze(x)
        cumulated = cumulated + val
    return np.squeeze(cumulated)

def normalize_histogram(hist):
    hist = hist / np.sum(hist)
    return hist


def calculate_BD(ori,gen):
    ori = cumulative_histogram(ori)
    gen = cumulative_histogram(gen)
    ori = normalize_histogram(ori)
    gen = normalize_histogram(gen)

    ori = ori.astype(np.float32)
    gen = gen.astype(np.float32)
    return cv2.compareHist(ori,gen,cv2.HISTCMP_BHATTACHARYYA)

# D1 of course (Histogram L1 distance)
def calculate_d1(ori, gen):
    ori = cumulative_histogram(ori)
    gen = cumulative_histogram(gen)
    ori = normalize_histogram(ori)
    gen = normalize_histogram(gen)

    distance = 0
    for i in range(len(ori)):
        distance += abs(ori[i]-gen[i])
    return distance

# D2 of course (Histogram L2 distance)
def calculate_d2(ori, gen):
    ori = cumulative_histogram(ori)
    gen = cumulative_histogram(gen)
    ori = normalize_histogram(ori)
    gen = normalize_histogram(gen)

    distance = 0
    for i in range(len(ori)):
        distance += pow(abs(ori[i]-gen[i]),2)
    distance = np.sqrt(distance)
    return distance

# D4 of course (Histogram quadratic distance)
def calculate_d4(ori, gen):
    ori = cumulative_histogram(ori)
    gen = cumulative_histogram(gen)
    ori = normalize_histogram(ori)
    gen = normalize_histogram(gen)

    d = len(ori)

    similarity_matrix = np.identity(d)
    for i in range(d):
        for j in range(d):
            similarity_matrix[i,j]=1-abs((i-j)/255)
    
    diff = ori - gen
    distance = np.abs(np.matmul(np.matmul(diff.T, similarity_matrix), diff))
    return np.sqrt(distance)

# D6 of course (Histogram Mahalanobis distance)
def calculate_d6(ori, gen):
    _ori = cumulative_histogram(ori)
    _gen = cumulative_histogram(gen)
    _ori = normalize_histogram(np.squeeze(_ori))
    _gen = normalize_histogram(np.squeeze(_gen))
    diff = _ori-_gen

    concat_matrix = np.concatenate((ori,gen),axis=0)
    concat_matrix = np.squeeze(concat_matrix)
    covariance_matrix = np.cov(concat_matrix.T)

    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    distance = np.abs(np.dot(np.matmul(diff,inv_covariance_matrix),diff))
    return np.sqrt(distance)