<h2>Tensorflow-Tiled-Image-Segmentation-Skin-Cancer (2024/06/04)</h2>

This is an experimental Tiled Image Segmentation project for Skin-Cancer based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1XKNJOfi2_n8ldmtgT_NIex4d0PTq92PR/view">
 Tiled-Skin-Cancer-ImageMask-Dataset-X.zip</a> 
<br>

We have already applied the Tiled Image Segmentation strategy to some UNet Image Segmentation Models.<br>

<li> <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API"> MultipleMyeloma </a></li>
<li> <a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Oral-Cancer">Oral-Cancer</a></li>
<li> <a href="https://github.com/sarah-antillia/Tensorflow-Tiled-ImageMask-Segmentation-Oral-Cancer">Oral-Cancer based on Tiledly-Splitted 
ImageMask-Dataset</a></li>
<br> 
This is the fourth example to apply the strategy to a segmentation model for Skin-Cancer.

As mentioned in <a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Skin-Cancer">Tiled-ImageMask-Dataset-Skin-Cancer</a>, the pixel-size of the original images and masks in validation and test dataset of 
ISIC Challenge Datasets 2017
<a href="https://challenge.isic-archive.com/data/">ISIC Challenge Datasets 2017</a> is very large from 1K to 6K, which is too large to use for a training of an ordinary segmentation model.<br>
Therefore, Tiled-Image-Segmentation method may be effective to infer the skin cancer regions for the large images.<br>
<br> 
In this experiment, we employed the following strategy:<br>
<b>
<br>
1. We trained and validated a TensorFlow UNet model using the Tiled-Skin-Cancer-ImageMask-Dataset, which was tiledly-split to 512x512 pixels.<br>
2. We applied the Tiled-Image Segmentation inference method to predict the segmentation regions for a test image 
with a resolution of 4K or 6K pixels. 
<br><br>
</b>  
 Please note that <a href="https://drive.google.com/file/d/1XKNJOfi2_n8ldmtgT_NIex4d0PTq92PR/view">
 Tiled-Skin-Cancer-ImageMask-Dataset-X.zip</a> contains two type of image and mask:
 <br> 
1. Tiledly-split to 512x512 image and mask  files.<br>
2. Size-reduced to 512x512 image and mask files.<br>

Namely, this is a mixed set of Tiled and Non-Tiled ImageMask Datasets.<br>
<hr>
Actual Tiled Image Segmentation for the images of 4K pixels.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td width="330" ><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/images/ISIC_0015155.jpg" width="320" height="auto"></td>

<td width="330" ><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0015155.png" width="320" height="auto"></td>
<!--
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/TCGA-CN-6019-01Z-00-DX1.0a01c44a-b7f4-427e-9854-fe2f0db631ab_1.jpg" width="320" height="auto"></td>
 -->
<td width="330"><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0015155.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td width="330" ><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/images/ISIC_0015224.jpg" width="320" height="auto"></td>

<td width="330" ><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0015224.png" width="320" height="auto"></td>
<!--
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/TCGA-CN-6019-01Z-00-DX1.0a01c44a-b7f4-427e-9854-fe2f0db631ab_1.jpg" width="320" height="auto"></td>
 -->
<td width="330"><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0015224.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Skin Cancer Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>


<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following web site.<br>
<pre>
ISIC Challenge Datasets 2017
https://challenge.isic-archive.com/data/
</pre>

<b>Citing 2017 datasets:</b>
<pre>
Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A.
 "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI),
  Hosted by the International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 [cs.CV]
</pre>
<b>License: CC-0</b><br>
<br>
See also:<br>

<a href="https://paperswithcode.com/dataset/isic-2017-task-1">ISIC 2017 Task 1</a><br>
<pre>
Introduced by Codella et al. in Skin Lesion Analysis Toward Melanoma Detection: 
A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), 
Hosted by the International Skin Imaging Collaboration (ISIC)
</pre>
<pre>
The ISIC 2017 dataset was published by the International Skin Imaging Collaboration (ISIC) as a large-scale dataset 
of dermoscopy images. The Task 1 challenge dataset for lesion segmentation contains 2,000 images for training with 
ground truth segmentations (2000 binary mask images).
</pre>
<br>


<br>

<h3>
<a id="2">
2 Skin Cancer ImageMask Dataset
</a>
</h3>
 If you would like to train this Skin-Cancer Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1XKNJOfi2_n8ldmtgT_NIex4d0PTq92PR/view">
 Tiled-Skin-Cancer-ImageMask-Dataset-X.zip</a> 
<br>

<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be

<pre>
./dataset
└─Skin-Cancer
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Tiled-Skin Cancer Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/_Tiled-Skin-Cancer-ImageMask-Dataset-X_.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is not necessarily large. Therefore, an online dataset augmentation 
strategy to train this Skin-Cancer model may be effective to get a better trained model.<br> 
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
4 Train TensorflowUNet Model
</h3>
 We have trained Skin-Cancer TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
, in which <b>generator</b> parameter setting in [model] section is <b>True</b> which enables to train TensorflowUNet model by the
online augmentor <a href="./src/ImageMaskAugmentor.py">ImageMaskAugmentor</a>.
<br>
<pre>
[model]
generator     = True
</pre>
 
Please move to ./projects/Skin-Cancer and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<pre>
; train_eval_infer.config
; 2024/06/03 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize= False

num_classes    = 1
base_filters   = 16
;base_kernels   = (7,7)
base_kernels   = (5,5)

num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.00004
clipvalue      = 0.5
;dilation       = (2,2)
dilation       = (1,1)

loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[dataset]
;Please specify a class name of your ImageDataset.
datasetclass   = "BaseImageMaskDataset"
color_order    = "bgr"

[train]
epochs         = 20
batch_size     = 2
steps_per_epoch  = 200
validation_steps = 100
patience       = 10
;metrics       = ["iou_coef", "val_iou_coef"]
metrics        = ["binary_accuracy", "val_binary_accuracy"]
model_dir      = "./models"
eval_dir       = "./eval"
image_datapath = "../../../dataset/Tiled-Skin-Cancer/train/images/"
mask_datapath  = "../../../dataset/Tiled-Skin-Cancer/train/masks/"
create_backup  = False

learning_rate_reducer = True
reducer_patience      = 4
save_weights_only     = True

;Inference execution flag on epoch_changed
epoch_change_infer     = True

; Output dir to save the infered masks on epoch_changed
epoch_change_infer_dir =  "./epoch_change_infer"

;Tiled-inference execution flag on epoch_changed
epoch_change_tiledinfer     = True

; Output dir to save the tiled-infered masks on epoch_changed
epoch_change_tiledinfer_dir =  "./epoch_change_tiledinfer"

; The number of the images to be inferred on epoch_changed.
num_infer_images       = 1


[eval]
image_datapath = "../../../dataset/Tiled-Skin-Cancer/valid/images/"
mask_datapath  = "../../../dataset/Tiled-Skin-Cancer/valid/masks/"

[test] 
image_datapath = "../../../dataset/Tiled-Skin-Cancer/test/images/"
mask_datapath  = "../../../dataset/Tiled-Skin-Cancer/test/masks/"

[infer] 
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output"
;merged_dir    = "./mini_test_output_merged"

[tiledinfer] 
overlapping = 128
images_dir    = "./mini_test/images/"
output_dir    = "./tiled_mini_test_output"
;merged_dir    = "./tiled_mini_test_output_merged"
bitwise_blending = False
;binarize      = True
;threshold  = 127


[segmentation]
colorize   = False
black      = "black"
white      = "green"
blursize   = None

[mask]
blur       = False
binarize   = True
threshold  = 127

[generator]
debug         = False
augmentation  = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [90, 180, 270]
shrinks  = [0.8]
shears   = [0.1]

deformation = True
distortion  = True

[deformation]
alpah    = 1300
sigmoid  = 8

[distortion]
gaussian_filter_rsigma= 40
gaussian_filter_sigma = 0.5
distortions           = [0.03,]
</pre>

In this configuration file above, we added the following parameters to enable <b>epoch_change_infer</b> and 
<b>epoch_change_tiledinfer</b> callbacks in [train] section.<br>
We added the following Python scripts to this repository for the epoch change inferences:<br>
<li> <a href="./src/EpochChangeInferencer.py">EpochChangeInferencer.py</a> </li>
<li> <a href="./src/EpochChangeTiledInferencer.py">EpochChangeTiledInferencer.py</a> </li>
<br>
By using these callbacks, on every epoch_change, the inference and tile-inference procedures can be called
 for an image in <b>mini_test</b> folder.<br><br>
<b>Epoch_change_inference</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<b>Epoch_change_tiled-inference</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/epoch_change_tiledinfer.png"  width="1024" height="auto"><br>
<br>
These inferred masks on_epch_change will be helpful to examine the parameters for training.<br>
<br>  
The training process has stopped at epoch 20.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/train_console_output_at_epoch_20.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
5 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Skin-Cancer</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Skin-Cancer.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/evaluate_console_output_at_epoch_20.png" width="720" height="auto">
<br><br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) and accuracy for this test dataset are very bad as shown below.<br>
<pre>
loss,0.8785
binary_accuracy,0.7546
</pre>

<h3>
6 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Skin-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Skin-Cancer.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
mini_test_images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/mini_test_images.png" width="1024" height="auto"><br>
mini_test_mask(ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<h3>
7 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Skin-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Skin-Cancer.<br>
<pre>
./4.infer_tiles.bat
</pre>
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer_aug.config
</pre>

<br>
Tiled inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/asset/tiled_mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>

<b>Enlarged Masks Comparison</b><br>
As shown below, the tiled-inferred-masks seem to be slightly clear than non-tiled-inferred-masks.<br>

<table>
<tr>
<th>Mask (ground_truth)</th>
<th>Non-tiled-inferred-mask</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0012223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0012223.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0012223.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0014936.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0014936.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0014936.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0014693.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0014693.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0014693.jpg" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0015224.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0015224.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0015224.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test/masks/ISIC_0015155.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0015155.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0015155.jpg" width="320" height="auto"></td>
</tr>
</table>
<br>
<br>
As shown below, the tiled-inferred-mask contains more detailed pixel level information than the non-tiled-inferred-mask.<br>
<br>
<table>
<tr>
<th>Non-tiled-inferred-mask</th>
<th>Tiled-inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/mini_test_output/ISIC_0015224.jpg" width="512" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Skin-Cancer/tiled_mini_test_output/ISIC_0015224.jpg" width="512" height="auto"></td>
</tr>
</table>
<br>
<br>

<hr>

<br>
<h3>
References
</h3>
<b>1. ISIC 2017 - Skin Lesion Analysis Towards Melanoma Detection</b><br>
Matt Berseth<br>
<pre>
https://arxiv.org/ftp/arxiv/papers/1703/1703.00523.pdf
</pre>

<b>2. ISIC Challenge Datasets 2017</b><br>
<pre>
https://challenge.isic-archive.com/data/
</pre>

<b>3. Skin Lesion Segmentation Using Deep Learning with Auxiliary Task</b><br>
Lina LiuORCID,Ying Y. Tsui andMrinal Mandal<br>
<pre>
https://www.mdpi.com/2313-433X/7/4/67
</pre>

<b>4. Skin Lesion Segmentation from Dermoscopic Images Using Convolutional Neural Network</b><br>
Kashan Zafar, Syed Omer Gilani, Asim Waris, Ali Ahmed, Mohsin Jamil, <br>
Muhammad Nasir Khan and Amer Sohail Kashif<br>
<pre>
https://www.mdpi.com/1424-8220/20/6/1601
</pre>

<b>5. Image-Segmentation-Skin-Lesion</b><br>
Toshiyuki Arai @antillia.com
<pre>
https://github.com/sarah-antillia/Image-Segmentation-Skin-Lesion
</pre>

<b>6. Tiled-ImageMask-Dataset-Skin-Cancer</b><br>
Toshiyuki Arai @antillia.com
<pre>
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Skin-Cancer
</pre>
