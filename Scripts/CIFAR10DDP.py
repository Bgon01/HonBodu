import os
import math
import numpy as np
from scipy.stats import levy_stable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ------------------------------------------------------------------------------
# Usage: python -m torch.distributed.launch --nproc_per_node=4 CIFAR10_DDP.py
# ------------------------------------------------------------------------------

# --------------------------
# Hyperparameters & Constants
# --------------------------
LEARNING_RATE = 0.1
CLIP = 10
NOISE_INIT = 0.005
NOISE_DECAY = 0.55
NUM_EPOCHS = 100

# --------------------------
# Custom Optimiser: AnnealisingOptimiser
# --------------------------
class AnnealisingOptimiser(Optimizer):
    """
    A heavy-tailed optimizer with annealed alpha:
    alpha = 2 - exp(-k * step)
    Adds heavy-tailed Lévy noise to updates, with momentum support.
    """
    def __init__(self, params, lr=1e-3, alpha=1.2, rho=0.05, decay=0.95,
                 noise_init=NOISE_INIT, noise_decay=NOISE_DECAY,
                 weight_decay=0, k=1/5, momentum=0.9):

        defaults = dict(lr=lr, alpha=alpha, rho=rho,
                        weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

        self.k = k
        self.decay = decay
        self.noise_init = noise_init
        self.noise_decay = noise_decay
        self.beta = 0  # symmetric noise
        self.num_steps = 1
        self.current_sigma = noise_init
        self.noise_samples = []

    def update_epoch(self):
        """Increment epoch counter (if needed externally)."""
        self.epoch += 1

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for loss computation.")
        closure = torch.enable_grad()(closure)
        loss = closure()

        for group in self.param_groups:
            group['alpha'] = 2 - math.exp(-self.k * self.num_steps)

        self.current_sigma = np.sqrt(self.noise_init) / ((1 + self.num_steps) ** (self.noise_decay / 2))
        total_elems = sum(p.numel() for group in self.param_groups for p in group['params'])
        big_noise = levy_stable.rvs(self.param_groups[0]['alpha'], self.beta,
                                    scale=self.current_sigma, size=total_elems)
        np.clip(big_noise, -CLIP, CLIP, out=big_noise)
        big_noise = torch.from_numpy(big_noise).float().to(self.param_groups[0]['params'][0].device)
        self.noise_samples.append(big_noise.detach().cpu().numpy())

        idx_start = 0
        for group in self.param_groups:
            lr = group['lr']
            momentum_val = group.get('momentum', 0)
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay'] != 0:
                    p.grad.data.add_(p.data, alpha=group['weight_decay'])

                # Momentum
                if momentum_val != 0:
                    buf = self.state.get(p, {}).get('momentum_buffer', p.grad.data.clone())
                    buf.mul_(momentum_val).add_(p.grad.data)
                    self.state[p]['momentum_buffer'] = buf
                    grad_update = buf
                else:
                    grad_update = p.grad.data

                elem_count = p.numel()
                noise_slice = big_noise[idx_start:idx_start + elem_count].view_as(p.data)
                idx_start += elem_count
                noise_coeff = (lr ** (1 / group['alpha'])) * noise_slice

                p.data.add_(grad_update, alpha=-lr)
                p.data.add_(noise_coeff)

        self.num_steps += 1
        return loss.item()

# --------------------------
# Training Loop
# --------------------------
def train_noise(model, train_loader, optimizer, epoch, train_losses, noise_list, device):
    model.train()
    running_loss = 0.0
    total_batches = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            return loss

        loss_val = optimizer.step(closure=closure)
        running_loss += loss_val
        total_batches += 1

    avg_loss = running_loss / total_batches
    train_losses.append(avg_loss)
    noise_list.append(optimizer.current_sigma)
    return avg_loss

# --------------------------
# Model Definition: ResNet18 for CIFAR-10
# --------------------------
def get_modified_resnet18():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# --------------------------
# Main Distributed Training Entry
# --------------------------
def main():
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # --------------------------
    # Data Loaders
    # --------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                       rank=dist.get_rank(), shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # --------------------------
    # Model, Optimizer, Scheduler
    # --------------------------
    model = get_modified_resnet18().to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer_anneal = AnnealisingOptimiser(
        model.parameters(), lr=LEARNING_RATE, alpha=1.7, rho=0.05,
        decay=0.95, noise_init=NOISE_INIT, noise_decay=NOISE_DECAY,
        k=1/5, weight_decay=5e-4, momentum=0.9
    )
    scheduler_anneal = optim.lr_scheduler.StepLR(optimizer_anneal, step_size=30, gamma=0.1)

    train_losses, noise_list = [], []

    # --------------------------
    # Training Loop
    # --------------------------
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        epoch_loss = train_noise(model, train_loader, optimizer_anneal, epoch,
                                 train_losses, noise_list, device)

        if dist.get_rank() == 0:
            model.eval()
            test_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, preds = outputs.max(1)
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()

            test_loss /= len(test_loader)
            test_acc = 100. * correct / total
            print(f"[Rank 0][Anneal] Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        scheduler_anneal.step()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
