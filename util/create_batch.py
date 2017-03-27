from __future__ import print_function

import os
import cv2
import sys
import time
from ctypes import *

import numpy as np
from sklearn.utils import shuffle

libc = CDLL("libc.so.6") # Linux


#############################
# Parameters 
BATCH_SIZE = 100
CHANNEL = 3
WIDTH = 224
HEIGHT =224
NUMBER_BATCH = 500
IMG_SHAPE = (CHANNEL, WIDTH, HEIGHT) # channel,height,width
train_val = "val"  # image folder
destination = "/home/ygao/gie_samples/samples/data/int8/googlenet/batches"


def create_batch(batch_file, labels, image_path, offset):

  """create batch file for sample_int8"""

  file_p = libc.fopen(batch_file, "wb")

  #############################
  # write batch/image info
  
  batch_info = [BATCH_SIZE, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]]
  # convert python list to C array
  c_batch_info = (c_int * len(batch_info))(*batch_info)
  c_batch_info_p = pointer(c_batch_info)
  # write batch info
  libc.fwrite(c_batch_info_p, sizeof(c_int), len(batch_info), file_p)
  print("Batch info is written. NCHW : ", BATCH_SIZE, IMG_SHAPE)

  #############################
  # write image
  number = 0
  print("offset: ", offset)
  batch_labels = labels[offset:offset+BATCH_SIZE]
  batch_path = image_path[offset:offset+BATCH_SIZE]
  batch_labels_f = []

  for path, label in zip(batch_path, batch_labels):

    if not os.path.exists(path):
      print("Missing file path: ", path, "  Skipped")
      continue

    if number == NUMBER_BATCH:
      break
    number += 1

    # collect labels
    batch_labels_f.append(float(label))

    img = cv2.imread(path)
    # resize image to Googlenet input size
    img = cv2.resize(img, (WIDTH, HEIGHT))
    # convert H*W*C to C*H*W
    img = np.transpose(img, [2, 0, 1])
    assert(img.shape==IMG_SHAPE)
    # convert to float
    img = img.astype(np.float32)
    # reshape to one dimension array
    img = img.flatten()
    img = img - 128.0
    img = img.tolist()

    # write image data to C array
    c_imgData = (c_float * len(img))(*img)
    c_imgData_p = pointer(c_imgData)

    # image data buffer size
    c_uncompSize = IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]

    # write image data
    nitems  = libc.fwrite(c_imgData_p, sizeof(c_float), c_uncompSize, file_p)

    assert(nitems == c_uncompSize)
  print("Image data is written. Image number: ", number)

  #############################
  # write labels
  c_batch_labels = (c_float * len(batch_labels))(*batch_labels_f)
  c_batch_labels_p = pointer(c_batch_labels)
  nlabels  = libc.fwrite(c_batch_labels, sizeof(c_float), BATCH_SIZE, file_p)
  assert(nlabels == BATCH_SIZE)
  print("Batch label is written")

  libc.fclose(file_p)

  statinfo = os.stat(batch_file)
  print(batch_file)
  print("Batch file size (in bytes) : ", statinfo.st_size)
  assert(statinfo.st_size == (4 + number + c_uncompSize * number) * 4)


def main():

  image_path = []
  labels = []

  # load image path and labels
  with open(train_val + ".txt") as fi:
    for line in fi:
      path, label = line.split(" ")
      path = os.path.join(train_val, path)

      image_path.append(path)
      labels.append(label)

  print("total images count :", len(image_path))
  print("total labels count : ", len(labels))

  # random shuffle data
  image_path, labels = shuffle(image_path, labels)
  assert(len(image_path) == len(labels))

  # generate batches
  offset = 0
  for i in range(NUMBER_BATCH):
    batch_file = os.path.join(destination,"batch" + str(i))
    t1 = time.time()
    create_batch(batch_file, labels, image_path, offset)
    t2 = time.time()
    print("Time:%.2f " % (t2 - t1))
    offset += BATCH_SIZE

  print("Done")

if __name__ == "__main__":
  main()

