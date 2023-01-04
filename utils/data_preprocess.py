from locale import ABDAY_1
from PIL import Image
import os
import torch
from torch.utils.data.dataset import Dataset
# import albumentations as A
from torchvision.transforms import transforms
# import cv2
import pickle


def get_train_val_set(trainset_path, valset_path):   
	# get all the images path and the corresponding labels

	with open(trainset_path, 'rb') as f:
		annot = pickle.load(f)
		train_image, train_label1 = annot

	with open(valset_path, 'rb') as f:
		annot = pickle.load(f)
		val_image, val_label1 = annot

	train_label = []
	for i in range(len(train_label1)):
		temp = []
		for j in range(68):
			x = train_label1[i][j][0]
			y = train_label1[i][j][1]
			temp.append(x)
			temp.append(y)
		train_label.append(temp)


	val_label = []
	for i in range(len(val_label1)):
		temp = []
		for j in range(68):
			x = val_label1[i][j][0]
			y = val_label1[i][j][1]
			temp.append(x)
			temp.append(y)
		val_label.append(temp)
	
	# Define your own transform here 
	# It can strongly help you to perform data augmentation and gain performance
	# ref: https://pytorch.org/vision/stable/transforms.html
	means = [0.485, 0.456, 0.406]
	stds = [0.229, 0.224, 0.225]
	train_transform = transforms.Compose([
							## TO DO ##
							# You can add some transforms here
							transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
							transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
							transforms.RandomGrayscale(p=0.2),
							# transforms.RandomHorizontalFlip(0.5),
							# transforms.RandomRotation(degrees=(-20, 20)),
                                                        
							# ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
							transforms.ToTensor(),
							
							# Transforms after conversion
							transforms.RandomErasing(p=0.25, scale=(0.08, 0.25), ratio=(0.8, 3.3), value=0, inplace=False),

							# experimental normalization for image classification 
							transforms.Normalize(means, stds),
						])
	# train_transform = A.Compose([
	# 	A.HorizontalFlip(p=0.5),
	# 	A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
	# 	A.RandomBrightnessContrast(p=0.2),
	# 	A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2)),
	# 	A.Rotate(limit=(-15,15), p=0.5),
	# 	#####
	# 	A.ToTensor(),
	# 	A.Normalize(means, stds),
	# ])
  
	# normally, we dont apply transform to test_set or val_set
	val_transform = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize(means, stds),
	])

  
	train_set, val_set = Prepare_dataset(images=train_image, labels=train_label,transform=train_transform, prefix="./dataset/train_data/synthetics_train"), \
						Prepare_dataset(images=val_image, labels=val_label,transform=val_transform, prefix="./dataset/train_data/aflw_val")

	return train_set, val_set


class Prepare_dataset(Dataset):
	def __init__(self, images, labels, transform, prefix):
		# It loads all the images' file name and correspoding labels here
		self.images = images 
		self.labels = labels 
		
		# The transform for the image
		self.transform = transform
		
		# prefix of the files' names
		self.prefix = prefix
	
		print(f'Number of images is {len(self.images)}')
	
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, idx):
		# You should read the image according to the file path and apply transform to the images
		# Use "PIL.Image.open" to read image and apply transform
		img_pth = os.path.join(self.prefix, self.images[idx])
		image = Image.open(img_pth).convert('RGB')
		if self.transform:
			image = self.transform(image)


		label = self.labels[idx]
		# print(label)
		return image, label
