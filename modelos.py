import torch
import torch.nn as nn
import torch.nn.functional as F


class GEInet(nn.Module):
    def __init__(self):
        super(GEInet, self).__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.fc3 = nn.Linear(1024)


class min2019(nn.Module):

    def __init__(self):
        super(min2019, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        '''self.flatt = nn.Flatten
        self.fc = nn.Linear(16 * 6 * 6, 120)'''
        #self.fc1 = nn.Linear(10816, 305)
        #self.fc1 = nn.Linear(128 * 13 * 13, 305)
        self.fc1 = nn.Linear(1 * 13 * 13 * 64, 305)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), 2, stride=2)
        x = F.max_pool2d(F.leaky_relu(self.conv4(x)), 2, stride=2)
        x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


if __name__ == "__main__":

    input = torch.randn(1, 1, 240, 240)

    torch.manual_seed(50)
    net = min2019()
    print(net)
    out = net(input)
    print(out[0][:8])

'''
    torch.manual_seed(50)
    m = nn.Sequential(
        nn.Conv2d(1, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(16, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(10816, 305),
        nn.ReLU()
    )
    print(m)
    output = m(input)
    print(output)
'''
