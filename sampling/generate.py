import torch
import torch.nn.functional as F
from model.transformer import TinyGPT
from utils.dataset import TextDataset

def sample(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits = model(idx)                 # (1, t, V)
        logits = logits[:, -1, :] / temperature  # (1, V)

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            min_logit = v[:, -1]
            logits = torch.where(
                logits < min_logit,
                torch.full_like(logits, -float("Inf")),
                logits,
            )

        probs = F.softmax(logits, dim=-1)   # (1, V)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_id], dim=1)            # (1, t+1)

    return idx

def main():
    # ─── Load tokenizer & vocab ─────────────────────────────
    ds = TextDataset("data/the-verdict.txt", seq_length=32, min_freq=1)
    vocab      = ds.tokenizer.str_to_int
    inv_vocab  = {i:s for s,i in vocab.items()}
    vocab_size = len(vocab)

    # ─── Build model and load checkpoint ──────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=8,
        num_layers=4,
        ff_hidden=512,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    checkpoint = torch.load("checkpoints/epoch5.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # ─── Prime prompt ─────────────────────────────────────────
    prompt = "I HAD always thought Jack Gisburn rather a cheap genius—though a good fellow enough—"
    idx = torch.tensor([ds.tokenizer.encode(prompt)], dtype=torch.long).to(device)

    # ─── Generate ─────────────────────────────────────────────
    out_idx = sample(model, idx, max_new_tokens=50, temperature=0.7, top_k=20)
    out_tokens = out_idx[0].tolist()
    generated = ds.tokenizer.decode(out_idx[0].tolist())
    print("\n=== Generated Text ===\n")
    print(generated)
    print("\n======================\n")

if __name__ == "__main__":
    main()
