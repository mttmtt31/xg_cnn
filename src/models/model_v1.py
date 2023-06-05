import torch.nn as nn

# Define the CNN model
class XGCNN(nn.Module):
    def __init__(self, dropout:float=0.0):
        super(XGCNN, self).__init__()
        # 80x160
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (5, 5)),
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU()
        ) #16 x 76 x 156
        self.pooling_1 = nn.MaxPool2d(2,2, return_indices=True) # -> 16 x 38 x 78

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU()
        ) # -> 32 x 36 x 76
        self.pooling_2 = nn.MaxPool2d(2,2, return_indices=True) # -> 32 x 18 x 38

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64 x 16 x 36
        self.pooling_3 = nn.MaxPool2d(2,2, return_indices=True) # -> 64 x 8 x 18

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1)),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU()
        ) # -> 64 x 8 x 18
        self.pooling_4 = nn.MaxPool2d(2,2, return_indices=True) # -> 64 x 4 x 9

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(256, 1),
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