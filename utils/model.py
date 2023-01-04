import torch
import torch.nn as nn
from torchvision import models

# our model's input: 3 channel
# our model's output: 68*2 = 136, for (x,y) points of 68 landmarks

class residual_block(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, layer):
        super(residual_block, self).__init__()
        
        self.conv = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        # self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        # self.conv3 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()
        self.layer = layer

    def convblock(self, x):
        # x = self.relu(self.conv(x))
        # x = self.relu(self.conv2(x))
        for i in range(self.layer-1):
            x = self.relu(self.conv(x))
        x = self.batchnorm(self.relu(self.conv(x)))
        return x
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        return x + self.convblock(x)


class Network(nn.Module):
    def __init__(self, output_class=136):
        super(Network, self).__init__()

        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                                       )
        # self.residual_block64 = residual_block(in_size=64, hidden_size=64, out_size=64, layer=3)
        
        self.conv64_128 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(128),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                                        )
        # self.residual_block128 = residual_block(in_size=128, hidden_size=128, out_size=128, layer=2)

        self.conv128_256 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(256),
                                        # nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
                                        )
        # self.residual_block256 = residual_block(in_size=256, hidden_size=256, out_size=256, layer=3)

        self.conv256_512 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(512),
                                        # nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
                                        )
        self.residual_block512 = residual_block(in_size=512, hidden_size=512, out_size=512, layer=3)

        self.lastMaxpool = nn.MaxPool2d(kernel_size=3, stride=3)

        # self.fc1 = nn.Sequential(nn.Linear(2048, 512),
                                 # nn.ReLU())
        self.fc2 = nn.Linear(512, output_class)

        # self.model = models.resnet18()
        # self.model.conv1=nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.fc=nn.Linear(self.model.fc.in_features, output_class)
    

    def forward(self, x):
        # print('testing')
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # # x = self.conv4(x)
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        # # print(x.shape)
        # x = self.fc(x)
        # x = self.fc_final(x)
        # x = self.model(x)
        x = self.stem_conv(x)
        # x = self.residual_block64(x)
        x = self.conv64_128(x)
        # x = self.residual_block128(x)
        x = self.conv128_256(x)
        # x = self.residual_block256(x)
        x = self.conv256_512(x)
        # x = self.residual_block512(x)
        # x = self.residual_block256(x)
        x = self.lastMaxpool(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        print(x.shape)
        # x = self.fc1(x)
        x = self.fc2(x)  
        return x