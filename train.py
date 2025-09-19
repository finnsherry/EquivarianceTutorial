"""
Run FashionMNIST classification.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import lietorch
from tqdm import tqdm
import sty
import numpy as np
from models import CNN, PDEGCNN
from time import perf_counter

# Model
MODELS = [CNN, PDEGCNN]
EPOCHS = 60
LR = 0.05
LR_GAMMA = 0.96
BATCH_SIZE = 256
WEIGHT_DECAY = 0.005


def test(model, device, test_loader):
    """
    Evaluate the model
    """
    model.eval()
    test_loss = []
    acc_score = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            test_loss.append(loss(output, y).item())
            y = y.cpu().view(-1)
            _, prediction = torch.max(output.cpu(), dim=1)
            acc_score.append((y == prediction).sum().item() / float(y.numel()))

    test_loss = np.mean(test_loss)
    acc_score = np.mean(acc_score)
    print("test_loss: ", test_loss)
    print("acc_score: ", acc_score)


def val(model, device, test_loader):
    """
    Evaluate the model
    """
    model.eval()
    val_loss = []
    acc_score = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            val_loss.append(loss(output, y).item())
            y = y.cpu().view(-1)
            _, prediction = torch.max(output.cpu(), dim=1)
            acc_score.append((y == prediction).sum().item() / float(y.numel()))

    val_loss = np.mean(val_loss)
    acc_score = np.mean(acc_score)
    print("val_loss: ", val_loss)
    print("acc_score: ", acc_score)


def train(model, device, train_loader, optimizer):
    """Train one epoch"""
    model.train()
    train_loss = []
    start = perf_counter()
    for x, y in tqdm(
        train_loader,
        desc="Training",
        dynamic_ncols=True,
        unit="batch",
    ):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        batch_loss = loss(output, y)
        train_loss.append(float(batch_loss.cpu().item()))
        batch_loss.backward()
        optimizer.step()
    end = perf_counter()
    print("train_loss: ", np.mean(train_loss))
    print("epoch time: ", float(end - start))


def loss(output, y):
    return F.cross_entropy(output, y)

def train_model(architecture):
    # instanciate model
    model = architecture().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)

    total_params = sum(p.numel() for p in model.parameters(recurse=True))

    print(
        f"Model: {architecture.__module__}."
        + sty.fg.li_green
        + f"{architecture.__name__}"
        + sty.rs.all
        + f" with {total_params} parameters"
    )

    start = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        print(
            sty.fg.white
            + sty.bg.li_blue
            + sty.ef.b
            + f"Epoch {epoch}/{EPOCHS}:"
            + sty.rs.all
        )
        train(model, device, train_loader, optimizer)
        val(model, device, val_loader)
        scheduler.step()
    test(model, device, test_loader)
    end = perf_counter()
    print("train time: ", float(end - start))
    torch.save(model.state_dict(), f"{architecture.__module__}.pth")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = Compose((ToTensor(), Normalize(0.5, 0.5)))
    train_set, val_set = random_split(FashionMNIST(".", train=True, transform=transforms, download=True), (0.88, 0.12))
    test_set = FashionMNIST(".", train=False, transform=transforms, download=True),

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=512,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
    )

    print(f"LieTorch v{lietorch.__version__}, PyTorch v{torch.__version__}")
    cc = torch.cuda.get_device_capability(device)
    print(
        f'Using device "{device}": {torch.cuda.get_device_name(device)} (compute capability {cc[0]}.{cc[1]})'
    )

    for model in MODELS:
        train_model(model)