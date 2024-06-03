# Copyright 2023 antillia.com Toshiyuki Arai
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

# TensorflowUNetTileInferencer.py
# 2023/06/08 to-arai
# 2024/04/22: Moved infer_tiles method in TensorflowModel to this class 
# 2024/06/03: Modified to use TiledInferencer class.

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"

import sys
import shutil
import numpy as np

import cv2
from PIL import Image
import glob
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from NormalizedImageMaskDataset import NormalizedImageMaskDataset

from TensorflowUNet import TensorflowUNet
from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet
from TensorflowTransUNet import TensorflowTransUNet
from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus
from TensorflowEfficientNetB7UNet import TensorflowEfficientNetB7UNet
#from TensorflowXceptionLikeUNet import TensorflowXceptionLikeUNet
from TensorflowUNetInferencer import TensorflowUNetInferencer

from TensorflowModelLoader import TensorflowModelLoader
from TiledInferencer import TiledInferencer

class TensorflowUNetTiledInferencer(TensorflowUNetInferencer):

  def __init__(self, config_file):
    super().__init__(config_file)

    print("=== TensorflowUNetTiledInferencer.__init__ config?file {}".format(config_file))
    #self.config = ConfigParser(config_file)
    self.images_dir = self.config.get(ConfigParser.TILEDINFER, "images_dir")
    self.output_dir = self.config.get(ConfigParser.TILEDINFER, "output_dir")
    self.tiledinfer_binarize =self.config.get(ConfigParser.TILEDINFER,   "binarize", dvalue=True) 
    print("--- tiledinfer binarize {}".format(self.tiledinfer_binarize))
    self.tiledinfer_threshold = self.config.get(ConfigParser.TILEDINFER, "threshold", dvalue=60)
    self.sharpening = self.config.get(ConfigParser.TILEDINFER, "sharpening", dvalue=False)

    # Create a UNetMolde and compile
    #model          = TensorflowUNet(config_file)
    ModelClass = eval(self.config.get(ConfigParser.MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))

    self.unet  = ModelClass(config_file) 
    print("--- self.unet {}".format(self.unet))
    self.model = self.unet.model
    print("--- self.model {}".format(self.model))
    self.model_loader = TensorflowModelLoader(config_file)

    # 2024/04/22 Load Model
    self.loader = TensorflowModelLoader(config_file)
    self.loader.load(self.model)

    if not os.path.exists(self.images_dir):
      raise Exception("Not found " + self.images_dir)

    self.tiled_inferencer = TiledInferencer(self.model, config_file)
    print("=== Created TiledInferencer")

  def infer(self):
    print("=== TensorflowUNetTiledInferencer call tilede_inferencer.infer()")
    self.tiled_inferencer.infer()
 
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
  
    inferencer = TensorflowUNetTiledInferencer(config_file)
    inferencer.infer()

  except:
    traceback.print_exc()
    

