import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import CNN, ResNet

config = {
            'conv0_channel': 16,
            'conv0_kernel_size': 9,
            'down_repeat': 1,
            'down_kernel_size': 5,
            'down_dropout': 0.1,
            'down_use_reflection': True,
            'down_norm': 'BatchNorm2d',
            'res_repeat': 2,
            'res_kernel_size': 5,
            'res_dropout': 0.1,
            'res_use_reflection': False,
            'res_norm': 'BatchNorm2d',
            'res_output_relu': True,
         }

class MNISTDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        #img = self.images[idx]
        #label = self.labels[idx]
        
        im = torch.tensor(img).unsqueeze(0)
        return (im / 255.0).type(torch.float)
    
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    images = np.load("images_test.npy")

    data = images

    dataset = MNISTDataset(data)

    model = ResNet(config)
    model.load_state_dict( torch.load("model.pth") )
    model = model.to(device)
    model.eval()
    
    
    predictions = []


    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            sample = sample.to(device).unsqueeze(0)
            pred = model(sample)
            pred = torch.argmax(pred).item()
            predictions.append(pred)
            
    df = pd.DataFrame({"ImageId": np.arange( len(predictions) ) + 1, "Label": predictions})
    df.to_csv("submission.csv", index=False)
            