import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.models.resnet


class ModelSmall(nn.Module):
    def __init__(self):
        super(ModelSmall, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m == self.fc1:
                # Xavier's initialization for first layer (ReLU activation)
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
            elif m == self.fc2:
                # Kaiming initialization for second layer (no activation after it)
                init.kaiming_uniform_(m.weight, nonlinearity='linear')
            m.bias.data.fill_(0.00)  # Initialize bias to 0 for both layers

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))  # ReLU after the first layer
        x = self.fc2(x)  # No activation after the second layer
        return x


class ModelMedium(nn.Module):
    def __init__(self):
        super(ModelMedium, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m == self.fc3:
                # Kaiming initialization for third layer (no activation after it)
                init.kaiming_uniform_(m.weight, nonlinearity='linear')
            else:
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
            m.bias.data.fill_(0.00)  # Initialize bias to 0 for all layers

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))  # ReLU after the first layer
        x = F.relu(self.fc2(x))  # ReLU after the second layer
        x = self.fc3(x)  # No activation after the third layer
        return x


class ConvNet(nn.Module):
    def __init__(self, dropout=0.25):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(128, 10)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.bias.data.fill_(0.00)
        elif isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.00)
        # if fc2 is the last layer, we can use the following initialization
        if m == self.layers[-1]:
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('linear'))

    def forward(self, x):
        x = self.layers(x)
        return x


class Resnet_FT(nn.Module):
    def __init__(self, freeze=True):
        super(Resnet_FT, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if freeze:
            # freeze all layers
            for param in self.resnet.parameters():
                param.requires_grad = False
        # change the last layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # check if the last layer is trainable
        # for name, param in self.resnet.named_parameters():
        #     print(f'{name}: {param.requires_grad}')

    def forward(self, x):
        batch_size = x.size(0)
        x = x.repeat(1, 3, 1, 1)  # repeat grayscale image to 3 channels
        assert x.shape == (batch_size, 3, 28, 28)
        x = self.resnet(x)

        return x
