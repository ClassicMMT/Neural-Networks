import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            # leaky relu can be better
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


device = "mps" if torch.mps.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="data/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
# Where fake images are going to go
writer_fake = SummaryWriter("runs/GAN_MNIST/fake")
writer_real = SummaryWriter("runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1- D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)  # noise is z
        fake = gen(noise)
        # See the docs for BCELoss to make this more clear
        disc_real = disc(real).view(-1)
        # this is log(D(real))
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        # need to detach it so gradients are not computed here
        # can also to lossD.backward(retain_graph=True) later
        disc_fake = disc(fake.detach()).view(-1)
        # this is the log(1 - D(G(z))) part
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### Train Generator min log(1 - D(G(z))) => max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        print(
            f"i: [{batch_idx + 1}/{len(loader)}], Epoch [{epoch + 1}/{num_epochs}], Loss D: {lossD}, Loss G: {lossG}, mps: {torch.mps.is_available()}\n"
        )

        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)

            writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)

            step += 1
