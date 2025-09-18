"""
Run FashionMNIST classification.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import lietorch
from tqdm import tqdm
import sty
import numpy as np

# Model
MODEL = lietorch.models.rotnist.GroupM2Classifier4
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


def train(model, device, train_loader, optimizer):
    """Train one epoch"""
    model.train()
    train_loss = []
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
    print("train_loss: ", train_loss)


def loss(output, y):
    return F.cross_entropy(output, y)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose((ToTensor(), Normalize(0.5, 0.5)))
    train_loader = DataLoader(
        FashionMNIST(".", train=True, transform=transforms, download=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        FashionMNIST(".", train=False, transform=transforms, download=True),
        batch_size=512,
        shuffle=False,
    )

    print(f"LieTorch v{lietorch.__version__}, PyTorch v{torch.__version__}")
    cc = torch.cuda.get_device_capability(device)
    print(
        f'Using device "{device}": {torch.cuda.get_device_name(device)} (compute capability {cc[0]}.{cc[1]})'
    )

    # instanciate model
    model = MODEL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)

    total_params = sum(p.numel() for p in model.parameters(recurse=True))

    print(
        f"Model: {MODEL.__module__}."
        + sty.fg.li_green
        + f"{MODEL.__name__}"
        + sty.rs.all
        + f" with {total_params} parameters"
    )

    for epoch in range(1, EPOCHS + 1):
        print(
            sty.fg.white
            + sty.bg.li_blue
            + sty.ef.b
            + f"Epoch {epoch}/{EPOCHS}:"
            + sty.rs.all
        )
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()
