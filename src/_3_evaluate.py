# Evaluate reconstruction , generate viz
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from _1_vae import VAE
from _0_data import get_dataloader

def evaluate_reconstruction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model = VAE(latent_dim=64).to(device)   #make sure latent dim matches training
    load_path = Path(__file__).parent / "vae_baseline.pth"
    model.load_state_dict(torch.load(load_path, weights_only=True))
    model.eval()
    
    # get small batch for viz
    _,_, test_loader = get_dataloader(batch_size=5)
    images, _ = next(iter(test_loader))
    images = images.to(device)
    
    # forward pass
    with torch.no_grad():
        reconstructed, _, _ = model(images)
        
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # side-by-side plot
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(images[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_reconstruction()
