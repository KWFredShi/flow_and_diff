import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import CNFVAE, cnf_vae_loss

# Set up device and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True)

input_dim = 28 * 28
model = CNFVAE(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training CNF-VAE on MNIST...")
for epoch in range(30):
    total_loss = 0
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        x_recon, mu, std = model(x)
        loss = cnf_vae_loss(x, x_recon, mu, std)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(mnist):.4f}")


def show_images(images, title):
    images = images.view(-1, 1, 28, 28).detach().cpu()
    grid = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(8, 3))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title}.png")

# Reconstruction visualization
x, _ = next(iter(loader))
x = x[:16].to(device).view(16, -1)
x_recon, _, _ = model(x)

show_images(x, "Original MNIST Digits")
show_images(x_recon, "Reconstructed Digits")

# Generate samples
def generate_samples(model, num_samples=16):
    z0 = torch.randn(num_samples, model.decoder_net[0].in_features).to(device)
    with torch.no_grad():
        zk = model.flow(z0)
        samples = model.decoder(zk)
    return samples

generated_samples = generate_samples(model, num_samples=16)
show_images(generated_samples, "Generated MNIST Digits")