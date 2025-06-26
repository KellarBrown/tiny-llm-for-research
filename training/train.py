import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from utils.dataset import TextDataset
from model.transformer import TinyGPT

def main():
    batch_size = 16
    seq_length = 32
    min_freq = 1
    learning_rate = 3e-4
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Loading dataset...")
    dataset = TextDataset("data/the-verdict.txt", seq_length=seq_length, min_freq=min_freq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset.tokenizer.str_to_int)
    print(f"Vocab size: {vocab_size}, device: {device}")


    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=8,
        num_layers=4,
        ff_hidden=512,
        max_seq_len=512,
        dropout=0.1
    ).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg = total_loss / 100
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {avg:.4f}")
                total_loss = 0            

        ckpt_path = f"checkpoints/epoch{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch} completed. Model saved to {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()       