import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import flow_matching_OT_displacement, OT_displacement_cond_loss
import os

# Set up device and data
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

dataset_name = "CIFAR10"  # Change to "MNIST" or "CIFAR10"
if dataset_name == "MNIST":
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
elif dataset_name == "CIFAR10":
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
input_dim = 32 * 32 * 3 if dataset_name == "CIFAR10" else 28 * 28

model = flow_matching_OT_displacement(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if False: #os.path.exists(f"{dataset_name}_model.pth"):
    model.load_state_dict(torch.load(f"{dataset_name}_model.pth"))
    print(f"Loaded model for {dataset_name}")
else:
    model.load_state_dict(torch.load(f"{dataset_name}_model.pth"))
    print(f"No pre-trained model found for {dataset_name}. Training from scratch.")
    print(f"Training Flow Matching with OT Displacement on {dataset_name}...")
    # Training loop
    epochs = 5000 if dataset_name == "MNIST" else 4000
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            # print(x.shape)
            x1 = x.view(x.size(0), -1).to(device)
            # print(x1.shape)
            t = torch.rand((x.size(0), 1), device=device)
            # print(t.shape)
            x0 = torch.randn_like(x1, device = device)
            sigma_min = torch.full((x.size(0), 1), 0.01, device=device)
            loss = OT_displacement_cond_loss(model, x0, x1, t, sigma_min)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataset):.6f}")
    torch.save(model.state_dict(), f"{dataset_name}_model.pth")
    print(f"Saved model for {dataset_name}")

def show_images(images, title):
    if dataset_name == "CIFAR10":
        images = images.view(-1, 3, 32, 32).detach().cpu()
    else:
        images = images.view(-1, 1, 28, 28).detach().cpu()

    
    images = torch.clamp(images, 0.0, 1.0)

    grid = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(8, 3))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title}.png")

def flow_matching_generate_samples(model, num_samples=16):
    x0 = torch.randn(num_samples, model.input_dim, device=device)  # Random initial samples
    with torch.no_grad():
        for t in torch.linspace(0, 1, 50):
            t = torch.full((num_samples,1), t.item(), device=device)
            x_t = torch.cat([x0, model.time_embedding(t)], dim=1)
            flow = model.flow_predictor(x_t)
            x0 = x0 + flow * (1 / 50)
    return x0

# Generate samples
generated_samples = flow_matching_generate_samples(model, num_samples=16)
if dataset_name == "MNIST":
    generated_samples = generated_samples.view(-1, 1, 28, 28)
    show_images(generated_samples, f"Generated {dataset_name} Digits")
elif dataset_name == "CIFAR10":
    generated_samples = generated_samples.view(-1, 3, 32, 32)
    show_images(generated_samples, f"Generated {dataset_name} Images")


