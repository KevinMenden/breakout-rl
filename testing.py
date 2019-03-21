import gym
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    The policy network for appoximation of the Q function
    """

    def __init__(self):
        super(DQN, self).__init__()
        self.n_actions = 2

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, self.n_actions)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        print(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# create environment
env = gym.make('Breakout-v0')
screen = env.render(mode='rgb_array')
env.close()
transform_image = T.Compose([T.ToPILImage(), T.Resize(50),T.Grayscale(), T.ToTensor()])
screen = transform_image(screen)
screen = screen.unsqueeze(0)
dqn = DQN()
dqn(screen)