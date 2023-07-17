from dataloader import read_bci_data
from torch import Tensor, device, cuda, no_grad
from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple
from functools import reduce
from collections import OrderedDict
from tqdm import tqdm
import sys
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt
from dataloader import read_bci_data

class EEGNet(nn.Module):
    def __init__(self, activation: nn.modules.activation, dropout: float = 0.25) -> None:
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1,51),
            stride=(1,1),
            padding=(0,25),
            bias=False
            ),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(2,1),
            stride=(1,1),
            groups=16,
            bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d( kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=dropout)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1,15),
            stride=(1,1),
            padding=(0,7),
            bias=False
            ),
            nn.BatchNorm2d(32),
            activation(),
            nn.AvgPool2d( kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=dropout)        
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, inputs: TensorDataset) -> Tensor:
        FirstResult = self.firstconv(inputs)
        SecondResult = self.depthwiseConv(FirstResult)
        ThirdResult = self.separableConv(SecondResult)
        return self.classify(ThirdResult)

class DeepConvNet (nn.Module):
    def __init__(self, activation: nn.modules.activation, dropout: float, 
                 NumOfLinear: int) -> None:
        super().__init__()

        self.conv25 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=25,
            kernel_size=(1,5),
            bias=False
            ),
            nn.Conv2d(
            in_channels=25,
            out_channels=25,
            kernel_size=(2,1),
            bias=False
            ),
            nn.BatchNorm2d(25),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=dropout)
        )

        self.conv50 = nn.Sequential(
            nn.Conv2d(
            in_channels=25,
            out_channels=50,
            kernel_size=(1,5),
            bias=False
            ),
            nn.BatchNorm2d(50),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=dropout)
        )

        self.conv100 = nn.Sequential(
            nn.Conv2d(
            in_channels=50,
            out_channels=100,
            kernel_size=(1,5),
            bias=False
            ),
            nn.BatchNorm2d(100),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=dropout)
        )

        self.conv200 = nn.Sequential(
            nn.Conv2d(
            in_channels=100,
            out_channels=200,
            kernel_size=(1,5),
            bias=False
            ),
            nn.BatchNorm2d(200),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=dropout)
        )

        self.flatten_size = 8600
        interval = round((98.0)/ NumOfLinear)
        LayerSize = 100
        features = [8600]

        while LayerSize > 2:
            features.append(LayerSize)
            LayerSize -= interval
        features.append(2)

        layers = [('flatten', nn.Flatten())]

        for idx, in_features in enumerate(features[:-1]):
            layers.append((f'linear_{idx}', nn.Linear(in_features=in_features, out_features=features[idx+1], bias=True)))

            if idx != len(features) - 2:
                layers.append((f'activation_{idx}', activation()))
                layers.append((f'dropout_{idx}', nn.Dropout(p=dropout)))

        self.classify = nn.Sequential(OrderedDict(layers))

    def forward(self, inputs: TensorDataset) -> Tensor:
        tmp_results = self.conv25(inputs)
        tmp_results = self.conv50(tmp_results)
        tmp_results = self.conv100(tmp_results)
        tmp_results = self.conv200(tmp_results)
        return self.classify(tmp_results)

def show_results(model: str, epochs: int, accuracy: Dict[str, dict], keys: List[str]) -> None:
    plt.figure(0)
    model_name = model
    if model == 'EEG':
        plt.title('Activation function comparison(EEGNet)')
    else:
        plt.title('Activation function comparison(DeepConvNet)')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    for type, acc in accuracy.items():
        for model in keys:
            plt.plot(range(epochs), acc[model], label=f'{model}_{type}')
            print(f'{model}_{type:<17}{max(acc[model]):.2f} %')
    
    tmp_str = f'foo_{model_name}.png'
    plt.legend(loc = 'lower right')
    plt.savefig(tmp_str)
    plt.show()
    
def train(Model: str, epochs: int, learning_rate: float, batch_size: int,
          optimizer: op, loss_function: nn.modules.loss, dropout: float, NumOfLinear: int,
          train_device: device, train_dataset: TensorDataset, test_dataset: TensorDataset) -> None:

    if Model == 'EEG':
        keys = {'EEG_ELU', 'EEG_ReLU', 'EEG_LeakyReLU'}
        models = {
            'EEG_ELU' : EEGNet(nn.ELU, dropout=dropout).to(train_device),
            'EEG_ReLU' : EEGNet(nn.ReLU, dropout = dropout).to(train_device),
            'EEG_LeakyReLU' : EEGNet(nn.LeakyReLU, dropout=dropout).to(train_device)
        }
    else:
        keys = {'Deep_ELU', 'Deep_ReLU', 'Deep_LeakyReLU'}
        models = {
            'Deep_ELU' : DeepConvNet(nn.ELU, dropout=dropout, NumOfLinear=NumOfLinear).to(train_device),
            'Deep_ReLU' : DeepConvNet(nn.ReLU, dropout=dropout, NumOfLinear=NumOfLinear).to(train_device), 
            'Deep_LeakyReLU': DeepConvNet(nn.LeakyReLU, dropout=dropout, NumOfLinear=NumOfLinear).to(train_device)
        }

    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in keys},
        'test': {key: [0 for _ in range(epochs)] for key in keys}
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    for key, model in models.items():
        model_optimizer = optimizer(model.parameters(), lr = learning_rate)
        for epoch in tqdm(range(epochs)):
            model.train()
            for data, label in train_loader:
                inputs = data.to(train_device)
                labels = label.to(train_device).long()
                pred_labels = model.forward(inputs=inputs)

                model_optimizer.zero_grad()

                loss = loss_function(pred_labels, labels)
                loss.backward()
                model_optimizer.step()

                accuracy['train'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
            
            accuracy['train'][key][epoch] = 100.0 * accuracy['train'][key][epoch] / len(train_dataset)

            model.eval()
            with no_grad():
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long()
                    
                    pred_labels = model.forward(inputs=inputs)

                    accuracy['test'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
                
                accuracy['test'][key][epoch] = 100.0 * accuracy['test'][key][epoch] / len(test_dataset)
        
        print()
        cuda.empty_cache()
    
    show_results(model = Model, epochs = epochs, accuracy=accuracy, keys=keys)


def main() -> None:
    Model = 'Deep' # 'DeepConvNet'
    Epochs = 150
    LearningRate = 0.01
    BatchSize = 64
    Optimizer = op.Adam
    LossFunc = nn.CrossEntropyLoss()
    Dropout = 0.5
    NumOfLinear = 1

    print(f'Model: {Model}\nEpochs: {Epochs}\nLearning Rate: {LearningRate}\nBatchSize: {BatchSize}\nDropout: {Dropout}')
    if Model == 'Deep':
        print(f'NumOfLinear: {NumOfLinear}\n')
    train_data, train_label, test_data, test_label = read_bci_data()

    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))

    train_device = device("cuda" if cuda.is_available() else "cpu")

    train(Model = Model, epochs = Epochs, learning_rate = LearningRate,
          batch_size = BatchSize, optimizer = Optimizer, loss_function = LossFunc,
          dropout = Dropout, NumOfLinear = NumOfLinear, train_device = train_device, 
          train_dataset = train_dataset, test_dataset = test_dataset)

if __name__ == '__main__':
    main()