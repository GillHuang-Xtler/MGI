
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid                                                    # input 1*3*32*32
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),  # 1*12*16*16
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),         # 1*12*8*8
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),         # 1*12*8*8
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # print(x)
        out = self.body(x)
        # print(out)
        out = out.view(out.size(0), -1)
        # print(out)
        out = self.fc(out)
        # print(out)
        return out


# Fully connected neural network
class FC2(nn.Module):
    def __init__(self, channel, input_size, hidden, num_classes):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(channel * input_size * input_size, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden , num_classes)

    def forward(self, x):
        # print(x.size())
        # print(x)        # 1*3*32*32
        out = x.view(x.size(0), -1)
        # out = self.relu(out)
        # print(out.size())
        # print(out)          # 1 * 3072
        out = self.fc1(out)
        # print(out.size())
        # out = self.relu(out)
        # print(out)                      # 1*256
        # print(out.size())
        out = self.fc2(out)
        # print(out.size())
        out = self.relu(out)
        # print(out)                      # 1*100
        return out