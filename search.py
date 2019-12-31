import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import argparse
from models import ResNet, CNN
#import wandb

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
    
    


def train_mnist(trial, batch_size, model_name, epochs):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #print("Running on: ", device)
    
    if model_name == "cnn":
        config = {
            "kernel_size": trial.suggest_int("kernel_size", 3, 5),
            "nonlin": trial.suggest_categorical("nonlin", [nn.ReLU, nn.Sigmoid]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "stride": trial.suggest_int("stride", 1, 2),
            "norm": trial.suggest_categorical("norm", [nn.InstanceNorm2d, nn.BatchNorm2d]),
            "start_channels": trial.suggest_categorical("start_channels", [8, 16, 32, 64])
            
        }
        model = CNN(config).to(device)
    elif model_name == "resnet":
        config = {
            "conv0_channel": trial.suggest_categorical("conv0_channel", [8, 16, 32]),
            "conv0_kernel_size": trial.suggest_categorical("conv0_kernel_size", [7, 9]),
            
            "down_repeat": trial.suggest_int("down_repeat", 1, 2),
            "down_kernel_size": trial.suggest_categorical("down_kernel_size", [3, 5]),
            "down_dropout": trial.suggest_categorical("down_dropout", [None, 0.1, 0.2, 0.3]),
            "down_use_reflection": trial.suggest_categorical("down_use_reflection", [True, False]),
            "down_norm": trial.suggest_categorical("down_norm", [None, "BatchNorm2d", "InstanceNorm2d"]),
            
            "res_repeat": trial.suggest_int("res_repeat", 2, 6),
            "res_kernel_size": trial.suggest_categorical("res_kernel_size", [3, 5]),
            "res_dropout": trial.suggest_categorical("res_dropout", [None, 0.1, 0.2, 0.3]),
            "res_use_reflection": trial.suggest_categorical("res_use_reflection", [True, False]),
            "res_norm": trial.suggest_categorical("res_norm", [None, "BatchNorm2d", "InstanceNorm2d"]),
            "res_output_relu": trial.suggest_categorical("res_output_relu", [True, False]),
            
            "lin_units": trial.suggest_int("lin_units", 100, 500)
        }
        model = ResNet(config).to(device)
    
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    data = list(zip(images, labels))

    train_data, test_data = train_test_split(data, test_size=0.2)

    train_dataset = MNISTDataset(train_data)
    test_dataset = MNISTDataset(test_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

    criterion = nn.CrossEntropyLoss()
    
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
    
    for epoch in range(epochs):
        
        model.train()
        for step, batch in enumerate(train_dataloader):
            data, target = batch
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        #print("STEP", step)
            
        acc = eval_model(model, test_dataloader)
        #print(acc)
        trial.report(acc, step=epoch)
        
        if trial.should_prune():
            #print("PRUNED")
            raise optuna.exceptions.TrialPruned()
    return eval_model(model, test_dataloader)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", type=int, help="Batch size", required=True)
    parser.add_argument("-e", type=int, help="Num epochs", required=True)
    parser.add_argument("-r", type=int, help="Num runs", required=True)
    args = parser.parse_args()
    
    batch_size = args.b
    epochs = args.e
    trials = args.r
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.PercentilePruner(25, n_warmup_steps=epochs // 3))
    study.optimize(lambda x: train_mnist(x, batch_size, "resnet", epochs), n_trials=trials, catch=(RuntimeError,ValueError))
    print(study.best_trial)
    print(study.best_value)
    print(study.best_params)