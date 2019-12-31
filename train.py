import torch
from torch import nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import uuid
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


wandb.init("Digit Recognizer", resume="digit_recognizer_" + str(uuid.uuid4()))


class MNISTDataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        #img = self.images[idx]
        #label = self.labels[idx]
        
        im = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(label)
        return (im / 255.0).type(torch.float), label.type(torch.long)

def save(model):
    torch.save(model.state_dict(), "model.pth")

def eval_model(model, test_dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_step, batch in enumerate(test_dataloader):
                data, target = batch
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                predicted = torch.argmax(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct / total

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    data = list(zip(images, labels))

    #train_data, test_data = train_test_split(data, test_size=0.2)

    train_dataset = MNISTDataset(data)
    #test_dataset = MNISTDataset(test_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
    #test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=32)

    model = ResNet(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()



    for epoch in range(1000):
            
            model.train()
            for step, batch in enumerate(train_dataloader):
                data, target = batch
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if step % 250 == 0:
                    wandb.log({"loss": loss.item()}, step=epoch * len(train_dataloader) + step)
            
            #if epoch % 2 == 0:
            #    acc = eval_model(model, test_dataloader)
            #    wandb.log({"accuracy": acc}, step=epoch * len(train_dataloader) + step)
            #    print("Epoch: {} Accuracy: {}".format(epoch, acc))
            
            save(model)