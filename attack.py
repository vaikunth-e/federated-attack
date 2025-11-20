# python attack.py --snap grads/round1_client0 --iters 2500 --lr 0.1

import os, glob, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

# build model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def load_snapshot(snap_dir):
    weights = torch.load(os.path.join(snap_dir, "weights.pt"), map_location="cpu")
    meta = torch.load(os.path.join(snap_dir, "meta.pt"), map_location="cpu")
    grads = {}
    for fn in glob.glob(os.path.join(snap_dir, "grad_*.pt")):
        key = os.path.basename(fn).replace("grad_", "").replace(".pt", "").replace("_", ".")
        grads[key] = torch.load(fn, map_location="cpu")
    return weights, grads, meta

# returns a dict name->tensor with requires_grad False (the targetss)
def flatten_grads(named_params):
    return {n: p.detach().clone() for n, p in named_params.items()}

# compute  L2 distance between current grads (from x_var,y_logits) and target grads
def grad_distance(model, target_grads, x_var, y_logits):
    # soft labels
    y_prob = F.softmax(y_logits, dim=1)
    logits = model(x_var)
    loss = -(y_prob * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    # compute gradients of loss w.r.t. model parameters, with graph
    params = [p for _, p in model.named_parameters()]
    grads_cur = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True
    )

    dist = 0.0 

    # walk through model params and their corresponding current grads
    for (name, _p), g_cur in zip(model.named_parameters(), grads_cur):
        if name not in target_grads:
            continue
        g_tgt = target_grads[name]
        dist = dist + F.mse_loss(g_cur, g_tgt, reduction="sum")

    return dist


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True, help="Path to grads/roundX_clientY snapshot dir")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--tv", type=float, default=0.0, help="Total variation weight (optional)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights, grads, meta = load_snapshot(args.snap)
    B = int(meta["batch_size"])

    # build model and load weights
    model = Net().to(device)
    model.load_state_dict(weights)

    # prepare target grads on device
    tgt = {k: v.to(device) for k, v in grads.items()}

    # initialize dummy inputs and label logits
    x_var = torch.randn(B, 1, 28, 28, device=device)
    x_var.requires_grad_(True)

    y_logits = torch.zeros(B, 10, device=device)
    y_logits.requires_grad_(True)


    opt = optim.Adam([x_var, y_logits], lr=args.lr)

    iters = args.iters
    for t in range(iters):
        opt.zero_grad()
        dist = grad_distance(model, tgt, x_var, y_logits)
        dist.backward()
        opt.step()

        if (t+1) % 50 == 0:
            print(f"[{t+1}/{iters}] grad-match loss = {dist.item():.4e}")


    # save the reconstruction
    x_clip = x_var.detach().cpu().clamp(0, 1)
    save_image(x_clip, os.path.join(args.snap, "recon.png"), nrow=int(math.sqrt(B)) if int(math.sqrt(B))**2==B else 8)
    # output inferred labels
    hard_labels = y_logits.detach().cpu().argmax(dim=1).tolist()
    with open(os.path.join(args.snap, "recon_labels.txt"), "w") as f:
        for i, lbl in enumerate(hard_labels):
            f.write(f"sample {i}: class {lbl}\n")
    print(f"Saved reconstruction to {args.snap}/recon.png")

if __name__ == "__main__":
    main()
