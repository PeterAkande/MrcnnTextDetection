from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from numpy import zeros
import math
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# define the prediction configuration
class PredictionConfig(Config):
    NAME = "PredictionConfig"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
from skimage.color import rgb2gray
"""Evaluate_model function gotten directly from machinelearningmastery.com"""
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        #print("Ap",AP)
        if math.isnan(AP):
           APs.append(0)
        else:
           APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP
from skimage.color import gray2rgb as gg
def test(image,cfg,model,Gh=False):
   
   Image=plt.imread(image)
   # Gh is set to true if preprocessing i.e thresholding is wanted...
   #but in the test conducted so far
   #it does somewhat poorly
   #on thresholded pictures
   #as the model was not trained
   # with thresholded images
   if Gh:
     img = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
     blur= cv2.GaussianBlur(img,(5,5),0)
     ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
     #convert image to 3 channels
     Image=gg(img)
   scaled_image = mold_image(Image, cfg)
   sample = expand_dims(scaled_image, 0)
   yhat = model.detect(sample, verbose=0)
   return yhat,Image
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='/content/drive/My Drive/', config=cfg)
model.load_weights('/content/drive/My Drive/handwritting_models/handwrittingconfig20200830T0341/mask_rcnn_handwrittingconfig_0011.h5', by_name=True)
train_mAP  = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
test_mAP = evaluate_model(test_set, model, cfg)
print(test_mAP)
import os
import cv2 as cv
Y="/content/drive/My Drive/Dooooo/"
k=os.listdir(Y)
print(k)

"""Perform prediction for all images in The mentioned directory"""
#poor code for plotting tho
for I in range(len(k)):
  Hi,image=test(Y+k[I],cfg,model, Gh=True)
  fig,ax=plt.subplots(1)
  yht=Hi[0]
  ax.imshow(image)
  from matplotlib.patches import Rectangle
  for box in yht['rois']:
    y1, x1, y2, x2 = box
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height,linewidth=2,edgecolor="r",facecolor="none")
    ax.add_patch(rect)
plt.show()