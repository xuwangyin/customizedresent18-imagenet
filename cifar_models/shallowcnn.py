import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN2(nn.Module):
  def __init__(self, output_size=10):
    super(ShallowCNN2, self).__init__()
    # Accepted input: BCHW, range [0, 1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
    self.fc1 = nn.Linear(256*4*4, 256)
    self.fc2 = nn.Linear(256, output_size)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = F.leaky_relu(self.conv3(x))
    x = F.leaky_relu(self.conv4(x))
    x = x.view(x.shape[0], -1)
    # x = F.dropout(x, p=0.4)
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x

class ShallowCNN1(nn.Module):
  def __init__(self, output_size=10):
    super(ShallowCNN1, self).__init__()
    # Accepted input: BCHW, range [0, 1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
    # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
    self.fc1 = nn.Linear(128*7*7, 256)
    self.fc2 = nn.Linear(256, output_size)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = F.leaky_relu(self.conv3(x))
    # x = F.leaky_relu(self.conv4(x))
    x = x.view(x.shape[0], -1)
    # x = F.dropout(x, p=0.4)
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x


class ShallowCNN0(nn.Module):
  def __init__(self, output_size=10):
    super(ShallowCNN0, self).__init__()
    # Accepted input: BCHW, range [0, 1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=2)
    self.fc1 = nn.Linear(4608, output_size)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    return x


class ShallowCNN0_1(nn.Module):
  def __init__(self, output_size=10):
    super(ShallowCNN0_1, self).__init__()
    # Accepted input: BCHW, range [0, 1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=2)
    self.fc1 = nn.Linear(4608, 256)
    self.fc2 = nn.Linear(256, output_size)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = x.view(x.shape[0], -1)
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x

class ShallowCNN0_2(nn.Module):
    def __init__(self):
        super(ShallowCNN0_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepCNN0(nn.Module):
    def __init__(self):
        super(DeepCNN0, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1) # 32x32
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1) # 16x16
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.fc1 = nn.Linear(256*4*4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.4)
        x = self.fc1(x)
        return x


class DeepCNN1(nn.Module):
    def __init__(self):
        super(DeepCNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1) # 32x32
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1) # 16x16
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.fc1 = nn.Linear(256*4*4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.flatten(x, 1)
        # x = F.dropout(x, 0.4)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x

class DeepCNN2(nn.Module):
  def __init__(self, output_size=1):
    super(DeepCNN2, self).__init__()
    conv3_params = dict(kernel_size=3, padding=1)
    # Accepted input: BCHW, range [0, 1]
    self.conv1 = nn.Conv2d(3, 32, **conv3_params)
    self.conv2 = nn.Conv2d(32, 64, **conv3_params)
    # self.conv3 = nn.Conv2d(64, 64, **conv3_params)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 8 * 8, 512)
    self.fc2 = nn.Linear(512, output_size)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    # x = self.pool(F.relu(self.conv3(x)))
    x = x.view(x.shape[0], -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
