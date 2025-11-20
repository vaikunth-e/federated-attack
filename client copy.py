# client.py
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

CLIENT_ID = os.environ.get("CLIENT_ID", "0")

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# data
def get_dataloader():
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

model = Net()
trainloader = get_dataloader()
os.makedirs("grads", exist_ok=True)

# flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.current_round = 0

    def get_parameters(self, config=None):  # ← accept config arg now
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        self.current_round += 1
        this_round = config.get("round", self.current_round)

        # make directories
        snap_dir = f"grads/round{this_round}_client{CLIENT_ID}"
        os.makedirs(snap_dir, exist_ok=True)

        # grab (just) one batch to log inversion
        x, y = next(iter(trainloader))
        x = x[0:1]
        y = y[0:1]
        import torchvision.utils as vutils

        # save original data for comparison
        vutils.save_image(x, f"grads/round{this_round}_client{CLIENT_ID}/original.png")
        torch.save(y, f"grads/round{this_round}_client{CLIENT_ID}/original.pt")

        optimizer.zero_grad()
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()

        # save grads for each parameter in this batch and the weights
        torch.save(self.model.state_dict(), f"{snap_dir}/weights.pt")
        meta = {"batch_size": int(x.size(0)), "round": int(this_round), "client_id": str(CLIENT_ID)}
        torch.save(meta, f"{snap_dir}/meta.pt")
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                torch.save(p.grad.detach().cpu(), f"{snap_dir}/grad_{name.replace('.', '_')}.pt")

        optimizer.step()

        print(f"[Client {CLIENT_ID}] Logged one-batch grads for inversion at {snap_dir}")
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(trainloader.dataset), {}

if __name__ == "__main__":
    print(f"Starting Flower client id={CLIENT_ID}, connecting to 127.0.0.1:8080")

    # convert numpy client -> client before start
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model).to_client(),
    )
