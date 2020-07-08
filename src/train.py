# Simple MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import dataset
import model
import engine

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
MAX_LOSS = 9999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_set, test_set = dataset.create_dataset(transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)

model = model.Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=1)

for epoch in range(1, EPOCHS + 1):
    train_loss = engine.train(model, device, train_loader, optimizer, epoch)
    test_loss = engine.test(model, device, test_loader)
    scheduler.step()
    if test_loss < MAX_LOSS:
        torch.save(model.state_dict(), "mnist_cnn.pt")
