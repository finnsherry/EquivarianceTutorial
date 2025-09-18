import torch.nn as nn
import lietorch.nn as lnn


class CNN(nn.Module):
    """LeNet-5"""

    def __init__(self, classes: int = 10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 120), nn.Tanh(), nn.Dropout(0.1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(84, classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class PDEGCNN(nn.Module):
    its = 1
    kernel_size = [5, 5, 5]
    alpha = 0.65
    classes = 10

    def __init__(self, classes: int = 10):
        super().__init__()

        self.classes = classes
        c = 16

        self.lift = nn.Sequential(
            lnn.LiftM2Cartesian(
                in_channels=1, out_channels=c, orientations=8, kernel_size=5
            ),
            nn.BatchNorm3d(c, track_running_stats=False),
            nn.ReLU(),
        )

        self.pde = nn.Sequential(
            lnn.CDEPdeLayerM2(c, c, self.kernel_size, self.its, self.alpha),
            lnn.SpatialResampleM2(size=(14, 14)),
            nn.Dropout3d(0.1),
            lnn.CDEPdeLayerM2(c, c, self.kernel_size, self.its, self.alpha),
            lnn.SpatialResampleM2(size=(5, 5)),
            lnn.MaxProjectM2(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(5 * 5, self.classes),
        )

    def forward(self, x):
        x = self.lift(x)
        x = self.pde(x).max(dim=(-2, -1))  # Single "invariant" per channel.
        x = self.fc(x)
        return x
