import torch.nn as nn

""" 
Like V1, but more complex fc lyaer
"""

# Define the CNN model
class XGCNN(nn.Module):
    def __init__(self, dropout:float=0.0):
        super(XGCNN, self).__init__()
        # 93x140
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (5, 5), padding=1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (5, 5), padding=1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU()
        ) #32x89x136
        self.pooling_1 = nn.MaxPool2d(2,2, return_indices=True) # -> 34x44x68

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5), padding=1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (5, 5), padding=1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64x40x64
        self.pooling_2 = nn.MaxPool2d(2,2, return_indices=True) # -> 64x20x32

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU()
        ) # -> 128,16,28
        self.pooling_3 = nn.MaxPool2d(2,2, return_indices=True) # -> 128x8x14

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (1, 1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 1)),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU()
        ) # -> 256x8x14
        self.pooling_4 = nn.MaxPool2d(2,2, return_indices=True) # -> 256x4x7

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layer_1(x)
        x, _ = self.pooling_1(x)
        x = self.conv_layer_2(x)
        x, _ = self.pooling_2(x)
        x = self.conv_layer_3(x)
        x, _ = self.pooling_3(x)
        x = self.conv_layer_4(x)
        x, _ = self.pooling_4(x)
        x = self.fc(x)

        return x