import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

torch.manual_seed(53113)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cuda" if  not use_cuda else "cpu")
# device = torch.device("cpu")
batch_size = test_batch_size = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(1, 20, 5, 1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),

            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, 2),
        # )
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x

import time

def train(model, device, train_loader, optimizer, epoch,count, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start = time.time()
        output = model(data)
        end = time.time()
        res.append(end-start)
        count = count+1
    return count
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()
        #     ))=
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

lr = 0.01
momentum = 0.5
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

epochs = 1
count = 0
res = []
for epoch in range(1, epochs + 1):
    count = train(model, device, train_loader, optimizer, epoch,count)
print("--------------------------------------",len(res),count)
print(sum(res)/len(res))
#     test(model, device, test_loader)
#
# save_model = True
# if (save_model):
#     torch.save(model.state_dict(),"mnist_cnn.pt")