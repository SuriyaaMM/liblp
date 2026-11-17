import torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.view(-1, 784))))

def train(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.ToTensor()                    # no normalize — matches the test script
    train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=256)

    model   = MNISTNet().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        print(f"epoch {ep+1}: test acc = {correct/total:.4f}")

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print("saved -> mnist_mlp.pt")
    return model

if __name__ == "__main__":
    train(epochs=5)