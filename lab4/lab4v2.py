from dataloader import RetinopathyLoader
import torch.nn as nn
import torch.optim as op
from torch import Tensor, device, cuda, no_grad, load, save
from torch.utils.data import TensorDataset, DataLoader
from torch import max as tensor_max
from torchvision import transforms
import torchvision.models as torch_models
import os
import sys

from typing import Optional, Type, Union, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm


class BasicBloack (nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: nn.Module = None) ->None:
        super(BasicBloack, self).__init__()
        self.activation = nn.ReLU(inplace = True)

        self.block = nn.Sequential(
            nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        residual = inputs
        outputs = self.block(inputs)
        
        residual = self.down_sample(inputs)
        outputs = self.activation(outputs+residual)
        return outputs

class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, down_sample: Optional[nn.Module] = None) -> None:
        super(BottleneckBlock, self).__init__()

        external_channels = out_channels * self.expansion
        self.activation = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels = out_channels, out_channels = external_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(in_channels = out_channels, out_channels = external_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(external_channels),
        )
        self.down_sample = down_sample

    def forward(self, inputs: TensorDataset) -> Tensor:
        residual = inputs
        outputs = self.block(inputs)
        
        residual = self.down_sample(inputs)
        outputs = self.activation(outputs+residual)
        return outputs

class ResNet(nn.Module):
    def __init__(self, model: str, block: Type[Union[BasicBloack, BottleneckBlock]], layers: List[int], pretrain: int):
        super(ResNet, self).__init__()
    
        if pretrain:
            pretrained_resnet = getattr(torch_models, model)(pretrained = True)
            self.conv_1 = nn.Sequential(
                getattr(pretrained_resnet, 'conv1'),
                getattr(pretrained_resnet, 'bn1'),
                getattr(pretrained_resnet, 'relu'),
                getattr(pretrained_resnet, 'maxpool'),
            )

            self.conv_2 = getattr(pretrained_resnet, 'layer1')
            self.conv_3 = getattr(pretrained_resnet, 'layer2')
            self.conv_4 = getattr(pretrained_resnet, 'layer3')
            self.conv_5 = getattr(pretrained_resnet, 'layer4')

            self.classify = nn.Sequential(
                getattr(pretrained_resnet, 'avgpool'),
                nn.Flatten(),
                nn.Linear(getattr(pretrained_resnet, 'fc').in_features, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=50, out_features=5),
            )
            del pretrained_resnet
        
        else:
            self.current_channels = 64
            self.conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 3,
                    out_channels = 64,
                    kernel_size = 7,
                    stride = 2,
                    padding = 3,
                    bias = False
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            )

            self.conv_2 = self.make_layer(block=block,
                                          num_of_blocks=layers[0],
                                          in_channels=64)
            self.conv_3 = self.make_layer(block=block,
                                          num_of_blocks=layers[1],
                                          in_channels=128,
                                          stride=2)
            self.conv_4 = self.make_layer(block=block,
                                          num_of_blocks=layers[2],
                                          in_channels=256,
                                          stride=2)
            self.conv_5 = self.make_layer(block=block,
                                          num_of_blocks=layers[3],
                                          in_channels=512,
                                          stride=2)

            self.classify = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features=512 * block.expansion, out_features=50),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=50, out_features=5),
            )
    
    def make_layer(self, block: Type[Union[BasicBloack, BottleneckBlock]], num_of_blocks: int, in_channels: int, stride: int = 1) -> nn.Sequential:
        down_sample = None
        if stride != 1 or self.current_channels != in_channels * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels = self.current_channels,
                        out_channels = in_channels * block.expansion,
                        kernel_size = 1,
                        stride = stride,
                        bias = False),
                nn.BatchNorm2d(in_channels * block.expansion)
            )
        
        layers = [
            block(
                in_channels = self.current_channels,
                out_channels = in_channels * block.expansion,
                stride = stride,
                down_sample = down_sample
            )
        ]

        self.current_channels = in_channels * block.expansion
        layers += [block(in_channels = self.current_channels, out_channels = in_channels) for _ in range(1, num_of_blocks)]

        return nn.Sequential(*layers)


    def forward(self, inputs: TensorDataset) -> Tensor:
        tmp_results = inputs
        for idx in range(1,6):
            tmp_results = getattr(self, f'conv_{idx}')(tmp_results)
        return self.classify(tmp_results)


def resnet18 (pretrain: int = 0) -> ResNet:
    return ResNet(model = 'resnet18', block = BasicBloack, layers = [2,2,2,2], pretrain = pretrain)

def resnet50 (pretrain: int = 0) -> ResNet:
    return ResNet(model = 'resnet50', block = BottleneckBlock, layers = [3,4,6,3], pretrain = pretrain)

def show_results(target_model: str, epochs: int, accuracy: Dict[str, dict], prediction: Dict[str, np.ndarray], ground_truth: np.ndarray, keys: List[str]) -> None:
    if not os.path.exists('./results'):
        os.mkdir('./results')
    
    plt.figure(0)
    plt.title(f'Results {target_model}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    # txt = 'results_accuracy.txt'
    # f = open(txt, 'w')

    for dataset, acc in accuracy.items():
        for model in keys:
            plt.plot(range(epochs), acc[model], label = f'{model}_{dataset}')
            print(f'{target_model}_{dataset}: {max(acc[model]):.2f} %')
            # print(f'{model}_{dataset}: {max(acc[model]):.2f} %', file = f)
    
    # f.close()

    plt.legend(loc = 'lower right')
    plt.tight_layout()
    
    plt.savefig(f'./results/{target_model}.png')
    plt.close()

    for key, pred_labels in prediction.items():
        cm = confusion_matrix(y_true = ground_truth, y_pred = pred_labels, normalize = 'true')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix ({key})')
        plt.tight_layout()
        
        plt.savefig(f'./results/{key.replace(" ", "_").replace("/","_")}_confusion.png')
        plt.close()

    
    # plt.show()

def train(target_model: str,  pretrain: int, batch_size: int, num_workers: int, learning_rate: float,
          epochs: int, optimizer: op, momentum: float, weight_decay: float, train_device: device, train_dataset: RetinopathyLoader, test_dataset: RetinopathyLoader) -> None:
    # print("begin")
    if target_model == 'ResNet18': 
        if pretrain:
            keys = ['ResNet18 (w/ pretraining)']
            models = {keys[0]: torch_models.resnet18(weights="DEFAULT").to(train_device)}
        else:
            keys = ['ResNet18 (w/o pretraining)']
            models = {keys[0]: torch_models.resnet18(weights=None).to(train_device)}

    else:
        if pretrain:
            keys = ['ResNet50 (w/ pretraining)']
            models = {keys[0]: torch_models.resnet50(weights="DEFAULT").to(train_device)}
        else:
            keys = ['ResNet50 (w/o pretraining)']
            models = {keys[0]: torch_models.resnet50(weights=None).to(train_device)}
    # print("show")
    
    accuracy = {
        'train': {key: [0 for _ in range(epochs)] for key in keys},
        'test': {key: [0 for _ in range(epochs)] for key in keys}
    }
    print("prediction")
    prediction = {key: None for key in keys}

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)
    # print("complete loading")
    ground_truth = np.array([], dtype=int)
    print("================ ground truth ================")
    for _ , label in tqdm(test_loader):
        ground_truth = np.concatenate((ground_truth, label.long().view(-1).numpy()))

    # stored_check_point = {
    #     'epoch': None,
    #     'model_state_dict': None,
    #     'optimizer_state_dict': None
    # }

    # print("check point")

    # last_epoch = checkpoint['epoch'] if not comparison and isload else 0
    
    for key, model in models.items():
        if optimizer is op.SGD:
            model_optimizer = optimizer(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
        else:
            model_optimizer = optimizer(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        # print("optimizer")
        # if not comparison and isload:
        #     model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        max_test_acc = 0
        # print("================ epochs ================")
        for epoch in range(epochs):
            model.train()
            print(f'======= epoch {epoch} =======\n')
            for data, label in tqdm(train_loader):

                inputs = data.to(train_device)
                labels = label.to(train_device).long().view(-1)

                pred_labels = model(inputs)

                model_optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(pred_labels, labels)
                loss.backward()
                model_optimizer.step()

                accuracy['train'][key][epoch] += (tensor_max(pred_labels, 1)[1] == labels).sum().item()
            accuracy['train'][key][epoch] = 100.0 * accuracy['train'][key][epoch] / len(train_dataset)
            # print("eval")
            model.eval()
            with no_grad():
                pred_labels = np.array([], dtype = int)
                # print("================ test loader ================")
                for data, label in test_loader:
                    inputs = data.to(train_device)
                    labels = label.to(train_device).long().view(-1)

                    outputs = model(inputs)
                    outputs = tensor_max(outputs, 1)[1]
                    # print("before concat")
                    pred_labels = np.concatenate((pred_labels, outputs.cpu().numpy()))
                    # print("after cocat")
                    accuracy['test'][key][epoch] += (outputs == labels).sum().item()
                
                accuracy['test'][key][epoch] = 100.0 * accuracy['test'][key][epoch] / len(test_dataset)

                if accuracy['test'][key][epoch] > max_test_acc:
                    max_test_acc = accuracy['test'][key][epoch]
                    prediction[key] = pred_labels

        print("\n")

        cuda.empty_cache()

    show_results(target_model=target_model, epochs = epochs, 
                 accuracy = accuracy, prediction = prediction, ground_truth = ground_truth, keys = keys)

def main() -> None:
    
    is18 = False

    model = 'ResNet18' if is18 else 'ResNet50'
    # if is18:
    #     model = 'ResNet18'
    # else:
    #     model = 'ResNet50'
    
    pretrain = 1
    batch_size = 4
    num_workers = 4
    learning_rate = 0.001
    epochs = 10  # 10 for res18 5 for res50
    optimizer = op.SGD
    weight_decay = 0.0005
    momentum = 0.9
    
    train_dataset = RetinopathyLoader('new_train', 'train')
    test_dataset = RetinopathyLoader('new_test', 'test')
    
    train_device = device("cuda" if cuda.is_available() else "cpu")
    # print(train_device)
    train(target_model=model, pretrain=pretrain, 
            batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate, epochs=epochs,
            optimizer=optimizer, momentum=momentum, weight_decay=weight_decay, train_device=train_device,
            train_dataset=train_dataset, test_dataset=test_dataset)

main()