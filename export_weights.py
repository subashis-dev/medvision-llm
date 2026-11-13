import os
import torch

ckpt_path = "checkpoints/medvision_best.pt"
out_path = "models/medvision_model.pt"

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    torch.save(
        {
            "model_state_dict": ckpt["model_state_dict"],
            "val_loss": ckpt.get("val_loss", 0.0),
            "epoch": ckpt.get("epoch", 0),
            "vocab_size": ckpt.get("vocab_size", 256),
            "d_model": ckpt.get("d_model", 256),
            "params": ckpt.get("params", 6396544),
        },
        out_path,
    )
    print(f"Exported inference weights: {out_path} ({os.path.getsize(out_path) / (1024**2):.1f} MB)")
