# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# EpochChangeTiledInferencer.py
# 2024/06/01:  

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import numpy as np

import shutil
import sys
import cv2
import glob
import traceback
import tensorflow as tf
from GrayScaleImageWriter import GrayScaleImageWriter
from MaskColorizedWriter import MaskColorizedWriter
from PIL import Image
from ConfigParser import ConfigParser

class EpochChangeTiledInferencer(tf.keras.callbacks.Callback):

  def __init__(self, model, config_file):
    self.model = model
    print("=== EpochChangeInferencer.__init__ config?file {}".format(config_file))
    self.config = ConfigParser(config_file)
    self.num_classes = self.config.get(ConfigParser.MODEL, "num_classes")

    self.images_dir = self.config.get(ConfigParser.INFER, "images_dir")

    self.output_dir    = self.config.get(ConfigParser.TRAIN, "epoch_change_tiledinfer_dir", dvalue="./epoch_change_tiledinfer")
    self.num_infer_images = self.config.get(ConfigParser.TRAIN, "num_infer_images", dvalue=1)


    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)

    self.colorize = self.config.get(ConfigParser.SEGMENTATION, "colorize", dvalue=False)
    self.black    = self.config.get(ConfigParser.SEGMENTATION, "black",    dvalue="black")
    self.white    = self.config.get(ConfigParser.SEGMENTATION, "white",    dvalue="white")
    self.blursize = self.config.get(ConfigParser.SEGMENTATION, "blursize", dvalue=None)
    self.writer   = GrayScaleImageWriter(colorize=self.colorize, black=self.black, white=self.white, verbose=False)

    self.maskcolorizer = MaskColorizedWriter(self.config, verbose=False)
    self.mask_colorize = self.config.get(ConfigParser.INFER, "mask_colorize", dvalue=False)
    self.MARGIN       = self.config.get(ConfigParser.TILEDINFER, "overlapping", dvalue=0)
    self.bitwise_blending    = self.config.get(ConfigParser.TILEDINFER, "bitwise_blending", dvalue=True)
    self.tiledinfer_binarize  = self.config.get(ConfigParser.TILEDINFER, "binarize", dvalue=False)
    self.tiledinfer_threshold = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=127)
    #print("--- bitwise_blending {}".format(self.bitwise_blending))
    self.bgcolor = self.config.get(ConfigParser.TILEDINFER, "background", dvalue=0)  

    self.color_order = self.config.get(ConfigParser.DATASET,   "color_order", dvalue="rgb")
   
    image_files  = glob.glob(self.images_dir + "/*.png")
    image_files += glob.glob(self.images_dir + "/*.jpg")
    image_files += glob.glob(self.images_dir + "/*.tif")
    image_files += glob.glob(self.images_dir + "/*.bmp")
    self.width        = self.config.get(ConfigParser.MODEL, "image_width")
    self.height       = self.config.get(ConfigParser.MODEL, "image_height")
    self.split_size  = self.config.get(ConfigParser.TILEDINFER, "split_size", dvalue=self.width)
    #print("---split_size {}".format(self.split_size))
    
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")
  
    num_images = len(image_files)
    if self.num_infer_images > num_images:
      self.num_infer_images =  num_images
    if self.num_infer_images < 1:
      self.num_infer_images =  1

    self.image_files = image_files[:self.num_infer_images]

    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)


  def on_epoch_end(self, epoch, logs):
    output_dir = self.output_dir
    expand     = True

    for image_file in self.image_files:
      image   = Image.open(image_file)
      #PIL image color_order = "rgb"
      w, h    = image.size

      # Resize the image to the input size (width, height) of our UNet model.      
      resized = image.resize((self.width, self.height))

      # Make a prediction to the whole image not tiled image of the image_file 
      cv_image= self.pil2cv(resized)
      predictions = self.predict([cv_image], expand=expand)
          
      prediction  = predictions[0]
      whole_mask  = prediction[0]    

      #whole_mask_pil = self.mask_to_image(whole_mask)
      #whole_mask  = self.pil2cv(whole_mask_pil)
      whole_mask  = self.normalize_mask(whole_mask)
      # 2024/03/30
      whole_mask  = self.binarize(whole_mask)
      whole_mask  = cv2.resize(whole_mask, (w, h), interpolation=cv2.INTER_NEAREST)     
      basename = os.path.basename(image_file)
      w, h  = image.size

      vert_split_num  = h // self.split_size
      if h % self.split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // self.split_size
      if w % self.split_size != 0:
        horiz_split_num += 1
      background = Image.new("L", (w, h), self.bgcolor)

      # Tiled image segmentation
      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = self.split_size * i
          upper = self.split_size * j
          right = left  + self.split_size
          lower = upper + self.split_size

          if left >=w or upper >=h:
            continue 
      
          left_margin  = self.MARGIN
          upper_margin = self.MARGIN
          if left-self.MARGIN <0:
            left_margin = 0
          if upper-self.MARGIN <0:
            upper_margin = 0

          right_margin = self.MARGIN
          lower_margin = self.MARGIN 
          if right + right_margin > w:
            right_margin = 0
          if lower + lower_margin > h:
            lower_margin = 0

          cropbox = (left  - left_margin,  upper - upper_margin, 
                     right + right_margin, lower + lower_margin )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          # Get the size of the cropped image.
          cw, ch  = cropped.size

          # Resize the cropped image to the model image size (width, height) for a prediction.
          cropped = cropped.resize((self.width, self.height))

          cvimage  = self.pil2cv(cropped)
          predictions = self.predict([cvimage], expand=expand)
          
          prediction  = predictions[0]
          mask        = prediction[0]    
          mask        = self.mask_to_image(mask)
          # Resize the mask to the same size of the corresponding the cropped_size (cw, ch)
          mask        = mask.resize((cw, ch))

          right_position = left_margin + self.width
          if right_position > cw:
             right_position = cw

          bottom_position = upper_margin + self.height
          if bottom_position > ch:
             bottom_position = ch

          # Excluding margins of left, upper, right and bottom from the mask. 
          mask         = mask.crop((left_margin, upper_margin, 
                                  right_position, bottom_position)) 
          #iw, ih = mask.size
          background.paste(mask, (left, upper))

      basename = os.path.basename(image_file)
      filename = "Epoch_" + str(epoch+1) + "_" + basename
      output_file = os.path.join(output_dir, filename)
      cv_background = self.pil2cv(background)

      bitwised = None
      if self.bitwise_blending:
        # Blend the non-tiled whole_mask and the tiled-background
        bitwised = cv2.bitwise_and(whole_mask, cv_background)

        bitwised = self.binarize(bitwised)
        cv2.imwrite(output_file, bitwised)
      else:
        # Save the tiled-background. 
        if self.tiledinfer_binarize:
          sharpened = self.binarize(cv_background)
          cv2.imwrite(output_file, sharpened)
        else:
          background.save(output_file)

   
  def predict(self, images, expand=True):
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    

  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

  
  def sharpen(self, image):
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel) 
    return sharpened
  
  def mask_to_image(self, data, factor=255.0, format="RGB"):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    image = image.convert(format)
    return image
  
  def normalize_mask(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    return data

  def binarize(self, mask):
    if self.num_classes == 1:
      #algorithm = cv2.THRESH_OTSU
      #_, mask = cv2.threshold(mask, 0, 255, algorithm)
      if self.tiledinfer_binarize:
        #algorithm = "cv2.THRESH_OTSU"
        #print("--- tiled_infer: binarize {}".format(algorithm))
        #algorithm = eval(algorithm)
        #_, mask = cv2.threshold(mask, 0, 255, algorithm)
        mask[mask< self.tiledinfer_threshold] =   0
        mask[mask>=self.tiledinfer_threshold] = 255
    else:
      pass
    return mask     

