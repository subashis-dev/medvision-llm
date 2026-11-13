import math
import os
import time
import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import MedVisionVLM
from tokenizer import MedicalTokenizer


def evaluate(model: nn.Module, val_loader, device: str, tok: MedicalTokenizer) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                _, loss = model(images, input_ids, targets=targets, loss_mask=loss_mask)

            mask_sum = loss_mask.sum().item()
            total_loss += loss.item() * mask_sum
            total_tokens += mask_sum

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss


def run_sample_generation(model: nn.Module, test_ds, device: str, tok: MedicalTokenizer):
    model.eval()
    print("\n" + "-" * 60)
    print("  🩺 REAL-TIME CLINICAL GENERATION ON TEST SCAN")
    print("-" * 60)

    # Test 3 different samples
    indices = [0, len(test_ds) // 2, len(test_ds) - 1]
    for idx in indices:
        sample = test_ds[idx]
        img = sample["images"].unsqueeze(0).to(device)
        cat = sample["category"]

        prompt = "Scan: Brain MRI | Task: Analyze clinical case and diagnose"
        prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
        prompt_ids.append(tok.sep_token_id)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = model.generate(
                img,
                prompt_tensor,
                max_new_tokens=140,
                temperature=0.3,
                eos_token_id=tok.eos_token_id,
            )

        gen_tokens = gen_ids[0].tolist()
        if tok.sep_token_id in gen_tokens:
            sep_idx = gen_tokens.index(tok.sep_token_id)
            resp_tokens = gen_tokens[sep_idx + 1 :]
        else:
            resp_tokens = gen_tokens

        resp_text = tok.decode(resp_tokens, skip_special=True)
        print(f"Ground Truth Pathology: [{cat.upper()}]")
        print(f"Generated Diagnosis   : {resp_text}\n")
    print("-" * 60)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"  🧠 TRAINING MEDVISION-LLM ON 10,287 SCANS + AUGMENTATION")
    if device == "cuda":
        print(f"  ⚡ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"=======================================================\n")

    batch_size = 32
    epochs = 5
    lr = 3e-4

    train_loader, val_loader, tok = get_dataloaders(
        batch_size=batch_size,
        img_size=128,
        max_seq_len=160,
    )

    model = MedVisionVLM(
        vocab_size=tok.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ffn=1024,
        max_seq_len=180,
        num_vis_tokens=16,
    ).to(device)

    ckpt_path = "checkpoints/medvision_best.pt"
    if os.path.exists(ckpt_path):
        print(f"Transferring weights from previous checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
    else:
        best_val_loss = float("inf")

    print(f"Model Architecture: {model.get_parameter_count():,} Parameters")
    print(f"Training on: {len(train_loader.dataset):,} Scans | Validation: {len(val_loader.dataset):,} Scans")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.5e-5)
    scaler = torch.amp.GradScaler("cuda")

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_tokens = 0

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                _, loss = model(images, input_ids, targets=targets, loss_mask=loss_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            mask_sum = loss_mask.sum().item()
            train_loss += loss.item() * mask_sum
            train_tokens += mask_sum

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                curr_loss = train_loss / max(1, train_tokens)
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(train_loader):03d}] "
                    f"Train Loss: {curr_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        epoch_train_loss = train_loss / max(1, train_tokens)

        # Validate
        val_loss = evaluate(model, val_loader, device, tok)
        val_ppl = math.exp(min(10.0, val_loss))
        epoch_time = time.time() - epoch_start

        print(
            f"\n>>> Epoch {epoch:02d} Complete ({epoch_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Perplexity: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "vocab_size": tok.vocab_size,
                    "d_model": 256,
                    "params": model.get_parameter_count(),
                    "total_scans": len(train_loader.dataset) + len(val_loader.dataset),
                },
                ckpt_path,
            )
            print(f"⭐ Saved new best checkpoint -> {ckpt_path} (Val Loss: {val_loss:.4f})")

        run_sample_generation(model, val_loader.dataset, device, tok)

    total_time = time.time() - total_start
    print(f"\n=======================================================")
    print(f"  🎉 TRAINING COMPLETE in {total_time/60:.2f} minutes!")
    print(f"  🏆 Best Validation Loss: {best_val_loss:.4f}")
    print(f"  💾 Saved Model: checkpoints/medvision_best.pt")
    print(f"=======================================================\n")


if __name__ == "__main__":
    train()
