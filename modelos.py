import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def features_loader(input):
    item = np.loadtxt(input, delimiter=',', dtype='float')
    return item


class ReID(nn.Module):

    def __init__(self):
        super(ReID, self).__init__()
        self.fc1 = nn.Linear(817, 305)
        self.fc2 = nn.Linear(305, 305)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ReID2(nn.Module):

    def __init__(self):
        super(ReID2, self).__init__()
        self.fc1 = nn.Linear(817, 305)
        self.fc2 = nn.Linear(305, 2048)
        self.fc3 = nn.Linear(2048, 305)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ReID3(nn.Module):

    def __init__(self):
        super(ReID3, self).__init__()
        self.fc1 = nn.Linear(817, 305)
        self.fc2 = nn.Linear(305, 2048)
        self.fc3 = nn.Linear(2048, 305)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x


class ReID4(nn.Module):

    def __init__(self):
        super(ReID4, self).__init__()
        self.fc1 = nn.Linear(817, 305)
        self.fc2 = nn.Linear(305, 2048)
        self.ln = nn.LayerNorm(2048)
        self.fc3 = nn.Linear(2048, 305)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x


class ReID5(nn.Module):

    def __init__(self):
        super(ReID5, self).__init__()
        self.norm1 = nn.LayerNorm(817)
        self.norm2 = nn.LayerNorm(305)
        self.fc1 = nn.Linear(817, 1024)
        self.fc2 = nn.Linear(1024, 305)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class ReID6(nn.Module):

    def __init__(self):
        super(ReID6, self).__init__()
        self.norm1 = nn.LayerNorm(817)
        self.fc1 = nn.Linear(817, 305)
        self.norm2 = nn.LayerNorm(305)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.norm2(x)
        return x


class min2019(nn.Module):

    def __init__(self):
        super(min2019, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1 * 13 * 13 * 64, 128)
        self.fc2 = nn.Linear(128, 305)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv4(x)), 2, stride=2)
        x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class min2019b(nn.Module):

    def __init__(self, classify=False, num_classes=None):
        super(min2019b, self).__init__()

        self.classify = classify
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        #self.fc1 = nn.Linear(1 * 13 * 13 * 64, 128)
        #self.logits = nn.Linear(128, self.num_classes)

        if self.num_classes is not None:
            self.logits = nn.Linear(1 * 13 * 13 * 64, 32)
            self.norm = nn.LayerNorm(32, elementwise_affine=False)
        else:
            self.logits = nn.Linear(1 * 13 * 13 * 64, 305)
            self.norm = nn.LayerNorm(305, elementwise_affine=False)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv4(x)), 2, stride=2)
        x = x.view(-1, self.num_flat_features(x))
        #x = self.fc1(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = self.logits(x)
            x = self.norm(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class gaitRec(nn.Module):

    def __init__(self, classify=False, num_classes=None):
        super(gaitRec, self).__init__()

        self.classify = classify
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        if self.num_classes is not None:
            self.logits = nn.Linear(64 * 28 * 28, 32)
        else:
            self.logits = nn.Linear(64 * 28 * 28, 305)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, stride=2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.logits(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# if __name__ == "__main__":

#     input = torch.randn(1, 1, 240, 240)

#     torch.manual_seed(50)
#     net = min2019()
#     print(net)
#     out = net(input)
#     print(out[0][:8])
