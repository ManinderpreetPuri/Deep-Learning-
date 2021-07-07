import collections
import csv
from pathlib import Path
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
to_tensor_transform = transforms.ToTensor()
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoTokenizer


# TODO Task 1b - Implement LesionDataset
#The __init__ function should have the following prototype
# is the directory path with all the image files
    # is the csv file with image ids and their corresponding labels
    #image = Image.open('ISIC_0024306.jpg')

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
   ])
    
class LesionDataset(Dataset):
  def __init__(self, img_dir, labels_fname, augment=False):
      
      self.img_dir = img_dir
      self.augment = augment
      self.labels_fname = pd.read_csv(labels_fname)
    
  def __len__(self):
        return len(self.labels_fname)

  def __getitem__(self, idx):
  
        image_id = self.labels_fname.iloc[idx,0]
        image = Image.open(os.path.join(self.img_dir, image_id +'.jpg')).convert("RGB")
        labels = self.labels_fname.drop(['image'], axis = 1)
        labels = np.array(labels)
        labels = np.argmax(labels, axis = 1)
        label = labels[idx]
        if self.augment:
           image = train_transforms(image)
        else:
           image = to_tensor_transform(image)
        return image, label

# TODO Task 1e - Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):



# TODO Task 2b - Implement TextDataset
#               The __init__ function should have the following prototype
#                   def __init__(self, fname, sentence_len)
#                   - fname is the filename of the cvs file that contains each
#                     news headlines text and its corresponding label.
#                   - sentence_len the maximum sentence length you want the
#                     tokenized to return. Any sentence longer than that should
#                     be truncated by the tokenizer. Any shorter sentence should
#                     padded by the tokenizer.
#                We will be using the pretrained 'distilbert-base-uncased' transform,
#                so please use the appropriate tokenizer for it. NOTE: You will need
#                to include the relevant import statement.

class TextDataset(Dataset):
  def __init__(self, fname, sentence_len):
      
      self.fname = pd.read_csv(fname)
      self.sentence_len = sentence_len
      texts = self.fname[2]
      labels = self.fname[0].tolist()

      tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
      self.vocab_size = tokenizer.vocab_size

      tokens = tokenizer(texts, truncation=True, padding=True, max_length= sentence_len)
      self.tokens = tokens["input_ids"]       
      self.labels = labels
    
  def __len__(self):
        return len(self.labels_fname)

  def __getitem__(self, idx):
        inputs = torch.tensor(self.tokens[idx], device=self.device)
        label = torch.tensor(self.labels[idx], device=self.device)
        
        return inputs, label
