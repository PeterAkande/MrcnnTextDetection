import json
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import numpy as np
from imgaug import augmenters as iaa
from mrcnn.utils import Dataset
"""
The model was trained using pretrained coco weights, link for download has been provided in readme.md
N.B: Results of this are not optimal as this was only a petty project and it was trained on only 250 images...,Relative to how much noisy or different forms of handwrittings

keras==2.1.3 was used
tensorflow 1.15.2 was used

drive has already been mounted using

from google.colab import drive
drive.mount("/content/drive/")"""

class HandwrittingDataset(Dataset):
    def __init__(self,path):
        super().__init__()
        with open(path,"r") as anno:
            self.annotations=json.load(anno)
        self.all_pics=["Pic_"+str(I)+".jpg" for I in range(len(list(self.annotations.keys())))]
    def load_dataset(self,dir,Train=True):
        self.add_class("dataset",1,"Pic")
        img_dir=dir+"pictures_1/"
        annots=dir+"pictures.json"
        for names in self.all_pics:
            image_id=names[4:-4]
            if Train and int(image_id)>274:
                continue
            if not Train and int(image_id)<274:
                continue
            img_path=img_dir+names
            self.add_image("dataset",image_id=image_id,path=img_path,pic_name=names)
    def load_mask(self,image_id):
        ids=[]
        info=self.image_info[image_id]
        name=info["pic_name"]
        details=self.annotations[name]
        h,w=details[0][0],details[0][1]
        annots=details[1:]
        mask=np.zeros([h,w,len(annots)],np.uint8)
        for i in range(len(annots)):
            ann=annots[i]
            k, l= ann[1], ann[3]
            n, o = ann[0], ann[2]
            mask[k:l, n:o, i]=1
            ids.append(self.class_names.index(ann[-1]))
        return mask,np.array(ids,np.int32)
    def image_reference(self,image_id):
        inf=self.image_info[image_id]
        path=inf["path"]
        return path
tk="/content/drive/My Drive/My_dataset/pictures_1.json"
ih="/content/drive/My Drive/My_dataset/"
train_set = HandwrittingDataset(tk)
train_set.load_dataset(ih, Train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

test_set = HandwrittingDataset(tk)
test_set.load_dataset(ih, Train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
class HandwrittingConfig(Config):
   NAME="HandwrittingConfig"
   NUM_CLASSES=1+1
   STEPS_PER_EPOCH=274
   IMAGE_PER_GPU=2
#weights directory in google drive
w="/content/drive/My Drive/mask_rcnn_coco.h5"
config=HandwrittingConfig()
#config.display()
model=MaskRCNN(mode="training",model_dir="/content/drive/My Drive/handwritting_models/",config=config)
#load pretrained weights
model.load_weights(w,by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
#define augumentation techniques
augumentation=iaa.Sequential([iaa.Sometimes(0.6,iaa.GaussianBlur(sigma=(0,0.4))),iaa.Flipud(0.2)),iaa.Fliplr(0.5)],random_order=True)
#train model
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20,layers="heads",augmentation=augumentation)