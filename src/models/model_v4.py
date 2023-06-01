import torch
import torch.nn as nn

""" 
Like V2, but for the new images (distance and angle)
"""


# Define the CNN model
class XGCNN(nn.Module):
    def __init__(self, dropout:float=0.0):
        super(XGCNN, self).__init__()
        # 93x140
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5, 5)),
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU()
        ) #16x96x96
        self.pooling_1 = nn.MaxPool2d(2,2, return_indices=True) # -> 16x48x48

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU()
        ) # -> 32x46x46
        self.pooling_2 = nn.MaxPool2d(2,2, return_indices=True) # -> 32x23x23

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64x21x21
        self.pooling_3 = nn.MaxPool2d(2,2, return_indices=True) # -> 64x10x10

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64x10x10
        self.pooling_4 = nn.MaxPool2d(2,2, return_indices=True) # -> 64x5x5

        self.unflatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5 + 2, 256),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, distance, angle):
        x = self.conv_layer_1(x)
        x, _ = self.pooling_1(x)
        x = self.conv_layer_2(x)
        x, _ = self.pooling_2(x)
        x = self.conv_layer_3(x)
        x, _ = self.pooling_3(x)
        x = self.conv_layer_4(x)
        x, _ = self.pooling_4(x)
        x = self.unflatten(x)
        x = torch.cat((x, distance.unsqueeze(1), angle.unsqueeze(1)), dim=1).float()

        x = self.fc(x)

        return x