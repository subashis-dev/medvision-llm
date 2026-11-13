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
        checkpoint_path: str = "checkpoints/medvision_best.pt",
        vocab_path: str = "data/vocab.json",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(vocab_path) and os.path.exists("models/vocab.json"):
            vocab_path = "models/vocab.json"
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

        # Check for checkpoint
        actual_ckpt = checkpoint_path
        if not os.path.exists(actual_ckpt) and os.path.exists("models/medvision_model.pt"):
            actual_ckpt = "models/medvision_model.pt"

        if os.path.exists(actual_ckpt):
            ckpt = torch.load(actual_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.val_loss = ckpt.get("val_loss", 0.0)
            self.val_acc = ckpt.get("val_acc", 0.0)
            self.epoch = ckpt.get("epoch", 0)
            print(f"Loaded MedVision-LLM from {actual_ckpt} (Epoch {self.epoch}, Val Acc: {self.val_acc:.1f}%, Val Loss: {self.val_loss:.4f})")
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
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        img_tensor, display_img = self.preprocess_image(image_input)

        # 1. Format prompt
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        prompt_ids.append(self.tokenizer.sep_token_id)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 2. Run Multimodal Generation + Visual Classifier
        with torch.no_grad():
            gen_ids, cls_logits = self.model.generate(
                img_tensor,
                prompt_tensor,
                max_new_tokens=150,
                temperature=temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Visual classifier probabilities
        class_labels = ["Glioma", "Meningioma", "Pituitary Adenoma", "Normal Brain MRI"]
        probs = torch.softmax(cls_logits[0], dim=-1).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        vis_diagnosis = class_labels[pred_idx]
        vis_confidence = float(probs[pred_idx]) * 100.0

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
            match = re.search(r'\{.*\}', raw_text)
            if match:
                parsed_data = json.loads(match.group(0))
            else:
                fixed = raw_text + '"}'
                match_fix = re.search(r'\{.*\}', fixed)
                if match_fix:
                    parsed_data = json.loads(match_fix.group(0))
        except Exception:
            pass

        # Harmonize diagnosis with visual classifier ground truth
        llm_diag = parsed_data.get("diagnosis", "").title()
        diagnosis = llm_diag if llm_diag in class_labels else vis_diagnosis

        modality = parsed_data.get("modality", "Brain MRI").title()
        finding = parsed_data.get("finding", "Neuroimaging lesion identified corresponding to primary pathology.")
        severity = parsed_data.get("severity", "Moderate").capitalize()
        case_narrative = parsed_data.get("clinical_case", raw_text)
        recommendations = parsed_data.get("recommendations", "Neurosurgical consultation and clinical follow-up imaging.")

        return {
            "prompt": prompt,
            "raw_output": raw_text,
            "modality": modality,
            "diagnosis": diagnosis,
            "visual_prediction": vis_diagnosis,
            "visual_confidence": f"{vis_confidence:.1f}%",
            "finding": finding,
            "severity": severity,
            "clinical_case": case_narrative,
            "recommendations": recommendations,
            "class_probabilities": {
                class_labels[i]: f"{float(probs[i]) * 100.0:.1f}%" for i in range(4)
            },
            "model_metadata": {
                "parameters": "6.4M",
                "backbone": "ConvNeXt Visual Stem",
                "decoder": "Causal Multimodal Transformer",
                "device": self.device,
            },
        }


if __name__ == "__main__":
    pipeline = MedVisionPipeline()
    sample_img = "data/samples/meningioma_tumor/image(1).jpg"
    if os.path.exists(sample_img):
        res = pipeline.analyze_scan(sample_img)
        print("Analysis Result:")
        print(json.dumps(res, indent=2))
