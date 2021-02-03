from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model,Sequential,Model
import cv2
import pandas as pd
from pprint import pprint
import os
import numpy as np
import pickle
import sklearn
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import time

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from tqdm import tqdm
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils

# import torch.optim as optim

import numpy as np
import glob
from PIL import Image
'''
# data loader
#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'imidx':imidx, 'image':img,'label':lbl}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'imidx':imidx,'image':image, 'label':label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label =sample['imidx'], sample['image'], sample['label']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample    



#u2net.py

class REBNCONV(nn.Module):
  def __init__(self,in_ch=3,out_ch=3,dirate=1):
    super(REBNCONV,self).__init__()

    self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
    self.bn_s1 = nn.BatchNorm2d(out_ch)
    self.relu_s1 = nn.ReLU(inplace=True)

  def forward(self,x):

    hx = x
    xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

    return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

  src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

  return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    super(RSU7,self).__init__()

    self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

    self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
    self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

    self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

    self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

  def forward(self,x):

    hx = x
    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = self.pool1(hx1)

    hx2 = self.rebnconv2(hx)
    hx = self.pool2(hx2)

    hx3 = self.rebnconv3(hx)
    hx = self.pool3(hx3)

    hx4 = self.rebnconv4(hx)
    hx = self.pool4(hx4)

    hx5 = self.rebnconv5(hx)
    hx = self.pool5(hx5)

    hx6 = self.rebnconv6(hx)

    hx7 = self.rebnconv7(hx6)

    hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
    hx6dup = _upsample_like(hx6d,hx5)

    hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

    return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    super(RSU6,self).__init__()

    self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

    self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
    self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

    self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

    self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

  def forward(self,x):

    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = self.pool1(hx1)

    hx2 = self.rebnconv2(hx)
    hx = self.pool2(hx2)

    hx3 = self.rebnconv3(hx)
    hx = self.pool3(hx3)

    hx4 = self.rebnconv4(hx)
    hx = self.pool4(hx4)

    hx5 = self.rebnconv5(hx)

    hx6 = self.rebnconv6(hx5)


    hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

    return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
      super(RSU5,self).__init__()

      self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

      self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
      self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

      self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
      self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

      self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
      self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

      self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

      self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

      self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
      self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
      self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
      self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

  def forward(self,x):

    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = self.pool1(hx1)

    hx2 = self.rebnconv2(hx)
    hx = self.pool2(hx2)

    hx3 = self.rebnconv3(hx)
    hx = self.pool3(hx3)

    hx4 = self.rebnconv4(hx)

    hx5 = self.rebnconv5(hx4)

    hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

    return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    super(RSU4,self).__init__()

    self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

    self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
    self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
    self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

    self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

    self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
    self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

  def forward(self,x):

    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx = self.pool1(hx1)

    hx2 = self.rebnconv2(hx)
    hx = self.pool2(hx2)

    hx3 = self.rebnconv3(hx)

    hx4 = self.rebnconv4(hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

    return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

  def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
    super(RSU4F,self).__init__()

    self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

    self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
    self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
    self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

    self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

    self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
    self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
    self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

  def forward(self,x):

    hx = x

    hxin = self.rebnconvin(hx)

    hx1 = self.rebnconv1(hxin)
    hx2 = self.rebnconv2(hx1)
    hx3 = self.rebnconv3(hx2)

    hx4 = self.rebnconv4(hx3)

    hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
    hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
    hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

    return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

  def __init__(self,in_ch=3,out_ch=1):
    super(U2NET,self).__init__()

    self.stage1 = RSU7(in_ch,32,64)
    self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage2 = RSU6(64,32,128)
    self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage3 = RSU5(128,64,256)
    self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage4 = RSU4(256,128,512)
    self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage5 = RSU4F(512,256,512)
    self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage6 = RSU4F(512,256,512)

    # decoder
    self.stage5d = RSU4F(1024,256,512)
    self.stage4d = RSU4(1024,128,256)
    self.stage3d = RSU5(512,64,128)
    self.stage2d = RSU6(256,32,64)
    self.stage1d = RSU7(128,16,64)

    self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
    self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
    self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
    self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

    self.outconv = nn.Conv2d(6,out_ch,1)

  def forward(self,x):

    hx = x

    #stage 1
    hx1 = self.stage1(hx)
    hx = self.pool12(hx1)

    #stage 2
    hx2 = self.stage2(hx)
    hx = self.pool23(hx2)

    #stage 3
    hx3 = self.stage3(hx)
    hx = self.pool34(hx3)

    #stage 4
    hx4 = self.stage4(hx)
    hx = self.pool45(hx4)

    #stage 5
    hx5 = self.stage5(hx)
    hx = self.pool56(hx5)

    #stage 6
    hx6 = self.stage6(hx)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


    #side output
    d1 = self.side1(hx1d)

    d2 = self.side2(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = self.side3(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = self.side4(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = self.side5(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = self.side6(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

    return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

  def __init__(self,in_ch=3,out_ch=1):
    super(U2NETP,self).__init__()

    self.stage1 = RSU7(in_ch,16,64)
    self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage2 = RSU6(64,16,64)
    self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage3 = RSU5(64,16,64)
    self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage4 = RSU4(64,16,64)
    self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage5 = RSU4F(64,16,64)
    self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

    self.stage6 = RSU4F(64,16,64)

    # decoder
    self.stage5d = RSU4F(128,16,64)
    self.stage4d = RSU4(128,16,64)
    self.stage3d = RSU5(128,16,64)
    self.stage2d = RSU6(128,16,64)
    self.stage1d = RSU7(128,16,64)

    self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
    self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

    self.outconv = nn.Conv2d(6,out_ch,1)

  def forward(self,x):

    hx = x

    #stage 1
    hx1 = self.stage1(hx)
    hx = self.pool12(hx1)

    #stage 2
    hx2 = self.stage2(hx)
    hx = self.pool23(hx2)

    #stage 3
    hx3 = self.stage3(hx)
    hx = self.pool34(hx3)

    #stage 4
    hx4 = self.stage4(hx)
    hx = self.pool45(hx4)

    #stage 5
    hx5 = self.stage5(hx)
    hx = self.pool56(hx5)

    #stage 6
    hx6 = self.stage6(hx)
    hx6up = _upsample_like(hx6,hx5)

    #decoder
    hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


    #side output
    d1 = self.side1(hx1d)

    d2 = self.side2(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = self.side3(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = self.side4(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = self.side5(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = self.side6(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

    return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


#u2net_test.py
#!rm -r '/content/drive/My Drive/Reminder/Data/Tiffin_results' 
#%cd '/content/drive/My Drive/Reminder/Data/'

#from data_loader import RescaleT
#from data_loader import ToTensor
#from data_loader import ToTensorLab
#from data_loader import SalObjDataset
#from model import U2NET # full size version 173.6 MB
#from model import U2NETP # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):

  ma = torch.max(d)
  mi = torch.min(d)
  dn = (d-mi)/(ma-mi)
  return dn

def save_output(image_name,pred,d_dir):

  predict = pred
  predict = predict.squeeze()
  predict_np = predict.cpu().data.numpy()

  im = Image.fromarray(predict_np*255).convert('RGB')
  img_name = image_name.split(os.sep)[-1]
  image = io.imread(image_name)
  imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

  pb_np = np.array(imo)

  aaa = img_name.split(".")
  bbb = aaa[0:-1]
  imidx = bbb[0]
  for i in range(1,len(bbb)):
      imidx = imidx + "." + bbb[i]

  imo.save(d_dir+imidx+'.png')
#!mkdir 'Tiffin_results'

def main(image_dir,prediction_dir,model_dir):

# --------- 1. get image path and name ---------
  model_name='u2net'#u2netp
  print(os.getcwd())
#  for i in tqdm(os.listdir('/content/U-2-Net/test_data/validation')):
  print("In the loop")
  image_dir = image_dir#'/content/U-2-Net/test_data/test_images')
  print(image_dir)
  prediction_dir = os.path.join(prediction_dir)#'/content/U-2-Net/test_data/', model_name + '_results/' +os.sep)
  ##prediction_dir = ('Tiffin_results'+os.sep)
  model_dir = os.path.join(model_dir)#'/content/drive/My Drive/Reminder/Pytorch', model_name + '.pth')
  print(np.array(model_dir))
  img_name_list = glob.glob(image_dir + os.sep + '*')
  #print(os.path.join('/content//U-2-Net/test_data', 'test_images'))

  # --------- 2. dataloader ---------
  #1. dataloader
  test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                      lbl_name_list = [],
                                      transform=transforms.Compose([RescaleT(320),
                                                                    ToTensorLab(flag=0)])
                                      )
  test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=1)

    # --------- 3. model define ---------
  if(model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
  elif(model_name=='u2netp'):
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3,1)
  net.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))
  if torch.cuda.is_available():
    net.cuda()
  net.eval()
  print("before inner loop")
  image_arr = [] #contain the output of the images form the u2net model  

    # --------- 4. inference for each image ---------
  for i_test, data_test in enumerate(test_salobj_dataloader):
    #print("In the innner loop")
    #print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
      inputs_test = Variable(inputs_test.cuda())
    else:
      inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    image_arr.append(pred)
    # save results to test_results folder
    if not os.path.exists(prediction_dir):
      #print("galat hora hai")
      os.makedirs(prediction_dir, exist_ok=True)
    save_output(img_name_list[i_test],pred,prediction_dir)
    
    del d1,d2,d3,d4,d5,d6,d7
  return image_arr
'''
from .classes import *
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Object_Detector():
  def __init__(self, model_path):
      self.__load_model(model_path)
      print('model loaded')

  def __load_model(self, model_path):
      self.detection_graph = tf.Graph()
      with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
              serialized_graph = fid.read()
              od_graph_def.ParseFromString(serialized_graph)
              tf.import_graph_def(od_graph_def, name='')

      config = tf.ConfigProto()
      config.gpu_options.allow_growth= True

      with self.detection_graph.as_default():
          self.sess = tf.Session(config=config, graph=self.detection_graph)
          self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
          self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
          self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
          self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
          self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

      # load label_dict
      self.label_dict = {1: 'fish'}
      
      # warmup
      self.detect_image(np.ones((600, 600, 3)))

  def detect_image(self, image_np, score_thr=0.5, print_time=False):
      image_w, image_h = image_np.shape[1], image_np.shape[0]
      arr = []
      # Actual detection.
      t = time.time()
      (boxes, scores, classes, num) = self.sess.run(
        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
        feed_dict={self.image_tensor: np.expand_dims(image_np, axis=0)})
      if print_time:
          print('detection time :', time.time()-t)
      # Visualization of the results of a detection.
      for i, box in enumerate(boxes[scores>score_thr]):
          top_left = (int(box[1]*image_w), int(box[0]*image_h))
          bottom_right = (int(box[3]*image_w), int(box[2]*image_h))
          cv2.rectangle(image_np, top_left, bottom_right, (0,255,0), 3)
          
          #-----------------------------------------Cropping image from final image---------------------------------------#
          #x,y,b,h = box[1],box[0],box[3],box[2]
          arr.append([top_left,bottom_right])
          #-----------------------------------------------------END--------------------------------------------------------#
          
          #----------------------------------------Make new images from boxes----------------------------------------------#

          #-----------------------------------------------------END--------------------------------------------------------#

          cv2.putText(image_np, self.label_dict[int(classes[0,i])], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

      return image_np,arr

# MODEL_PATH = 'fish_ssd_fpn_graph/frozen_inference_graph.pb'
MODEL_PATH = 'results/frozen_inference_graph.pb'
object_detector = Object_Detector(MODEL_PATH)

def final():

  source_model = load_model('results/epoch_19550.h5',compile = False)
  model = Sequential()
  model= Model(inputs = source_model.input, outputs = source_model.get_layer('activation_2').output)   

  image_dir = 'media/images'
  prediction_dir = 'results/u2net_results/'
  model_dir = 'results/u2net/u2net.pth'
  #image_arr = main(image_dir,prediction_dir,model_dir) segmentaion code commented


  model_family = pickle.load(open('results/knn_model_family.pickle',"rb"))
  model_genus = pickle.load(open('results/knn_model_genus.pickle',"rb"))
  model_species = pickle.load(open('results/knn_model_species.pickle',"rb"))
  
  prediction_family,prediction_genus,prediction_species = [],[],[]
  
  for img in os.listdir(image_dir):
    
    name_of_image = img 
    image_path = os.path.join(image_dir,img)
    print(image_path)
    img = embedding(image_path,model)
    image= img.reshape([-1,512])
    prediction_family.append(model_family.predict(image))
    for i in range(5):print("")
    print("The prediccitons are such as of model_family ",model_family.predict(image))
    for i in range(5):print("")
    prediction_genus.append(model_genus.predict(image))
    prediction_species.append(model_species.predict(image))
    y_train_family = pickle.load(open("results/y_train_family.pickle","rb"))
    y_train_genus = pickle.load(open("results/y_train_genus.pickle","rb"))
    y_train_species = pickle.load(open("results/y_train_species.pickle","rb")) 
    print("The family,genus and species of the image is {},{},{}".format(model_family.predict(image),model_genus.predict(image),model_species.predict(image)))
    family = y_train_family[model_family.predict(image)[0]]
    genus = y_train_genus[model_genus.predict(image)[0]]
    species = y_train_species[model_species.predict(image)[0]]
    print("The species and genus is {}, {}, {}".format(family,species,genus))

  d = pd.read_csv('results/wt_and_cost.csv')
  for i in range(len(d)):
    if d.iloc[i,1] == family:avg_weight = d.iloc[i,2]

  final_result = {
                  'family':family,
                  'genus':genus,
                  'fishType':species,
                  'quantity':'5',
                  'weight':avg_weight,
                  'cost':'$100'
                 } 

  final_array = []
  final_array.append(final_result)

  #Detected fish image
  return final_array

def text_block(text_list, start_coord, spacing = 30, bg_color = 'white', block_shape = (256, 256,3), font = cv2.FONT_HERSHEY_SIMPLEX, thickness = 3, text_color = (0,0,255)):
  start_coord = list(start_coord)
  if bg_color == 'white':
    blank = np.ones(block_shape, dtype=np.uint8)*255
  else:
    blank = np.zeros(block_shape, dtype=np.uint8)*255

  coord = start_coord
  for text in text_list:
    blank = cv2.putText(blank, text, tuple(coord), font, 1, text_color, thickness, cv2.LINE_AA)
    coord[1] += spacing
  
  #cv2.imwrite('test.png', blank)
  return blank 

img = cv2.imread('media/images/out.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_,arr = object_detector.detect_image(img, score_thr=0.2)
plt.figure(figsize=(20, 10))
plt.imshow(img_)
plt.savefig('results/detected_fishes/out.png') 

for index,box in enumerate(arr):
  print(box[0][0])
  print(box[0][1])
  print(box[1][0])
  print(box[1][1])
  img = cv2.imread('media/images/out.jpg')
  image = img[box[0][1]:box[1][1],box[0][0]:box[1][0]]
  #shape = cv2.resize(image,(256,256))
  #blank_image = 255 * np.ones(shape=[256, 256, 3], dtype=np.uint8)
  #image = cv2.copyMakeBorder( image, image.shape[0]*2,image.shape[1]*2,image.shape[0]*2,image.shape[1]*2, cv2.BORDER_CONSTANT,value = [255,255,255])
  #imag = cv2.putText(blank_image, dic[0]['family']+'\n'+dic[0]['genus']+'\n'+dic[0]['fishType'], box[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
  
  ############################The image is getting classified###################
  model_family = pickle.load(open('results/knn_model_family.pickle',"rb"))
  model_genus = pickle.load(open('results/knn_model_genus.pickle',"rb"))
  model_species = pickle.load(open('results/knn_model_species.pickle',"rb"))

  y_train_family = pickle.load(open("results/y_train_family.pickle","rb"))
  y_train_genus = pickle.load(open("results/y_train_genus.pickle","rb"))
  y_train_species = pickle.load(open("results/y_train_species.pickle","rb")) 

  source_model = load_model('results/epoch_19550.h5',compile = False)
  model = Sequential()
  model= Model(inputs = source_model.input, outputs = source_model.get_layer('activation_2').output)   

  img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
  image = img.reshape(1, 256, 256, 3)
  image = model.predict(image)
  #img = embedding(image,model)
  image= image.reshape([-1,512])

  family = y_train_family[model_family.predict(image)[0]]
  genus = y_train_genus[model_genus.predict(image)[0]]
  species = y_train_species[model_species.predict(image)[0]]
  print("The species and genus is {}, {}, {}".format(family,species,genus))

  d = pd.read_csv('results/wt_and_cost.csv')
  for i in range(len(d)):
    if d.iloc[i,1] == family:avg_weight = d.iloc[i,2]

  final_result = {
                  'family':family,
                  'genus':genus,
                  'fishType':species,
                  'quantity':'5',
                  'weight':avg_weight,
                  'cost':'$100'
                 } 

  #####################################END##########################################
  blank_image = text_block([final_result['family'], final_result['genus'], final_result['fishType'],final_result['quantity'],\
    final_result['weight'],final_result['cost']], (20,20))
  #concatenating blank_image and image
  blank_image = blank_image.reshape([256,256,3])
  print("The type of blank image is {} and of img is {}".format(type(blank_image),type(img)))
  final_image = cv2.hconcat([img,blank_image])
  print("The index is ",index)
  cv2.imwrite('results/detected_fishes/'+str(index)+'.png',final_image)

  

  