import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 8, 3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
           # nn.Printer(),
            nn.Flatten(),
            nn.Linear(128*10*14, 7)      
        )
        
    
    def forward(self, x):
      x = self.seq(x)
      return x

# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.
def construct_resnet18():
    resnet18 = models.resnet18(pretrained=True)
    AB = 0
    for child in resnet18.children():
      AB += 1
      if AB < 4:
        for param in child.parameters():
          param.requires_grad = False
    
    resnet18.fc = nn.Linear(512, 7)

    return resnet18

    

# TODO Task 1f - Create your own models
class DropoutCovnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 8, 3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.4),
           # nn.Printer(),
            nn.Flatten(),
            nn.Linear(128*10*14, 7)      
        )

    def forward(self, x):
      x = self.seq(x)
      return x

# TODO Task 2c - Complete TextMLP
class TextMLP(nn.Module):
    def __init__(self, vocab_size, sentence_len, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size//2),
            nn.Flatten(),
            # ....
        )


# TODO Task 2c - Create a model which uses distilbert-base-uncased
#                NOTE: You will need to include the relevant import statement.
# class DistilBertForClassification(nn.Module):
#   ....