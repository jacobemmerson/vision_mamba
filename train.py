
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from src.ViM import ViM

SAVE_PATH = "./model.pth"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5
DROPOUT = 0.7
LAYERS = 12
REG = 0

## Loading ImageNet
CIFAR10_data = torchvision.datasets.CIFAR10('./data',
                                            train = True,
                                            transform = transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(CIFAR10_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=2)

## Defining Model Parameters
model = ViM(
    input_dim=128,  
    state_dim=64,
    d_conv=4,
    num_classes=10,  
    image_size=32,  
    patch_size=4,  
    channels=3,  
    dropout=DROPOUT,
    num_blocks=LAYERS, 
)


### Training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = REG)

model.train()
for epoch in range(EPOCHS):
    for i, (batch) in enumerate(data_loader):

        images, labels = batch

        # send tensor to the device
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backprop
        loss.backward()
        optimizer.step()
        model.zero_grad()

        if i % 20 == 0:
            print(f"EPOCH: {epoch} STEP: {i} LOSS: {loss.item():.4f}")

torch.save(model.state_dict(), SAVE_PATH)