#train VAE + track reconstruction and KL losses
import torch
import torch.optim as optim
import torch.nn.functional as F
from vae import VAE
from data import get_dataloader

def train():
    
    # SETUP --
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataloader = get_dataloader(batch_size=128)
    
    # TRAINING --
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        
        for data, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # forward pass
            recon_batch, mu, logvar = model(data)
            
            # calc losses
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        # Summary!
        num_samples = len(dataloader.dataset)
        print(f"Epoch {epoch+1} | "
              f"Total: {total_loss/num_samples:.2f} | "
              f"Recon: {total_recon/num_samples:.2f} | "
              f"KL: {total_kl/num_samples:.2f}")

if __name__ == "__main__":
    train()
