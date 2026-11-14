#train VAE + track reconstruction and KL losses
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from _1_vae import VAE
from _0_data import get_dataloader

# beta scales the KL term (Higgins et al., 2017) -> beta sweep passes in its own values + file names
def train(beta=0.1, save_name="vae_baseline.pth", plot_name="training_losses.png", show=True):

    # SETUP --
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=64).to(device)       #make sure latent dim matches in evaluate.py
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, val_loader, test_loader = get_dataloader(batch_size=128)

    # TRACK LOSSES --
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_losses = []
    val_recon_losses = []
    val_kl_losses = []


    # TRAINING --

    #beta weight on KL penalty is now a train() argument (default 0.1)

    # epochs = 10         # can adjust this for longer training
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # forward pass
            recon_batch, mu, logvar = model(data)
            
            # calc losses
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + (beta * kl_loss) 
            
            # backward pass
            loss.backward()
            optimizer.step()

            # track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        # TRAINING AVERAGES --

        num_samples = len(train_loader.dataset)

        train_total = total_loss / num_samples
        train_recon = total_recon / num_samples
        train_kl = total_kl / num_samples


        # VALIDATION --

        val_total , val_recon, val_kl= 0,0,0

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)

                # forward pass
                recon_batch, mu, logvar = model(data)

                # calc losses
                recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                val_total += recon_loss.item() + (beta * kl_loss.item())

        # VALIDATION AVERAGES --

        val_samples = len(val_loader.dataset)

        val_total /= val_samples
        val_recon /= val_samples
        val_kl /= val_samples
            
        # Summary!
        print(
            f"Epoch {epoch+1} | "
            f"Train Total: {train_total:.2f} | "
            f"Val Total: {val_total:.2f} | "
            f"Train Recon: {train_recon:.2f} | "
            f"Val Recon: {val_recon:.2f} | "
            f"Train KL: {train_kl:.2f} | "
            f"Val KL: {val_kl:.2f}"
)

        # STORE LOSSES --

        train_losses.append(train_total)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)

        val_losses.append(val_total)
        val_recon_losses.append(val_recon)
        val_kl_losses.append(val_kl)

    # SAVE MODEL --
    save_path = Path(__file__).parent / save_name
    torch.save(model.state_dict(), save_path)


    # PLOT LOSSES --

    epochs_range = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # total loss
    axes[0].plot(epochs_range, train_losses, label="Train")
    axes[0].plot(epochs_range, val_losses, label="Validation")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss per Sample")
    axes[0].legend()

    # reconstruction loss
    axes[1].plot(epochs_range, train_recon_losses, label="Train")
    axes[1].plot(epochs_range, val_recon_losses, label="Validation")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss per Sample")
    axes[1].legend()

    # KL divergence
    axes[2].plot(epochs_range, train_kl_losses, label="Train")
    axes[2].plot(epochs_range, val_kl_losses, label="Validation")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL per Sample")
    axes[2].legend()

    plt.tight_layout()
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / plot_name
    plt.savefig(plot_path, dpi=300)
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    train()