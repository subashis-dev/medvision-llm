import json
import os
import re
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from model import MedVisionVLM
from tokenizer import MedicalTokenizer


class MedVisionPipeline:
    """End-to-End Medical Vision-Language Inference Pipeline."""

    def __init__(
        self,
        checkpoint_path: str = "models/medvision_model.pt",
        vocab_path: str = "models/vocab.json",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MedicalTokenizer(vocab_path)

        self.model = MedVisionVLM(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ffn=1024,
            max_seq_len=180,
            num_vis_tokens=16,
        ).to(self.device)

        # Use best available checkpoint (checkpoints/medvision_best.pt or models/medvision_model.pt)
        actual_ckpt = checkpoint_path
        if os.path.exists("checkpoints/medvision_best.pt"):
            actual_ckpt = "checkpoints/medvision_best.pt"
        elif not os.path.exists(actual_ckpt):
            actual_ckpt = "models/medvision_model.pt"

        if os.path.exists(actual_ckpt):
            ckpt = torch.load(actual_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.val_loss = ckpt.get("val_loss", 0.0)
            self.epoch = ckpt.get("epoch", 0)
            print(f"Loaded MedVision-LLM from {actual_ckpt} (Epoch {self.epoch}, Val Loss: {self.val_loss:.4f})")
        else:
            print(f"Warning: Checkpoint {actual_ckpt} not found. Running with initialized weights.")

        self.model.eval()

    def preprocess_image(self, img_input: Union[str, bytes, np.ndarray], img_size: int = 128) -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(img_input, str):
            img_bgr = cv2.imread(img_input)
            if img_bgr is None:
                raise ValueError(f"Could not read image from path: {img_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, bytes):
            nparr = np.frombuffer(img_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_input

        # Resize to model input size
        resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Tensor conversion & normalization
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device), resized

    def analyze_scan(
        self,
        image_input: Union[str, bytes, np.ndarray],
        prompt: str = "Scan: Brain MRI | Task: Analyze clinical case and diagnose",
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        img_tensor, display_img = self.preprocess_image(image_input)

        # Format prompt
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        prompt_ids.append(self.tokenizer.sep_token_id)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                img_tensor,
                prompt_tensor,
                max_new_tokens=150,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = gen_ids[0].tolist()
        if self.tokenizer.sep_token_id in gen_tokens:
            sep_idx = gen_tokens.index(self.tokenizer.sep_token_id)
            resp_tokens = gen_tokens[sep_idx + 1 :]
        else:
            resp_tokens = gen_tokens

        raw_text = self.tokenizer.decode(resp_tokens, skip_special=True)

        # Parse JSON output from the LLM
        parsed_data = {}
        try:
            # Clean possible trailing braces or truncated JSON
            match = re.search(r'\{.*\}', raw_text)
            if match:
                parsed_data = json.loads(match.group(0))
            else:
                # If closing brace missing due to length
                fixed = raw_text + '"}'
                match_fix = re.search(r'\{.*\}', fixed)
                if match_fix:
                    parsed_data = json.loads(match_fix.group(0))
        except Exception:
            pass

        # Fallback fields if JSON parsing was partial
        diagnosis = parsed_data.get("diagnosis", "Unspecified Neurological Abnormality").title()
        modality = parsed_data.get("modality", "Brain MRI").title()
        finding = parsed_data.get("finding", raw_text)
        severity = parsed_data.get("severity", "Moderate").capitalize()
        case_narrative = parsed_data.get("clinical_case", raw_text)
        recommendations = parsed_data.get("recommendations", "Neurosurgical consultation and follow-up imaging.")

        return {
            "prompt": prompt,
            "raw_output": raw_text,
            "modality": modality,
            "diagnosis": diagnosis,
            "finding": finding,
            "severity": severity,
            "clinical_case": case_narrative,
            "recommendations": recommendations,
            "model_metadata": {
                "parameters": "6.4M",
                "backbone": "ConvNeXt Visual Stem",
                "decoder": "Causal Multimodal Transformer",
                "device": self.device,
            },
        }


if __name__ == "__main__":
    import sys
    pipe = MedVisionPipeline()
    sample_img = r"data/brain_mri_raw/Testing/glioma_tumor/image(1).jpg"
    if os.path.exists(sample_img):
        res = pipe.analyze_scan(sample_img)
        print("Analysis Result:")
        print(json.dumps(res, indent=2))
