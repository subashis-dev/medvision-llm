import os
import random
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import MedicalTokenizer

PROMPT_TEMPLATES = [
    "Scan: Brain MRI | Task: Analyze clinical case and diagnose",
    "Scan: Brain MRI | Query: What pathology is visible in this scan?",
    "Scan: Brain MRI | Question: Explain the clinical case and what happened",
    "Scan: Brain MRI | Clinical Request: Provide diagnosis, severity, and recommendations",
    "Scan: Brain MRI | Task: Complete neuro-radiology evaluation",
]

DIAGNOSTIC_CASES = {
    "glioma_tumor": [
        {
            "finding": "infiltrative heterogeneous intra-axial mass with surrounding vasogenic edema",
            "diagnosis": "glioma",
            "severity": "high",
            "case": "patient presented with focal neurological deficit and new-onset seizures. mri reveals cerebral parenchymal neoplasm with local mass effect and ventricular effacement.",
            "rec": "urgent neurosurgical consult for stereotactic biopsy, surgical resection, and oncology staging.",
        },
        {
            "finding": "poorly marginated high-signal mass lesion with significant perilesional edema and mass effect",
            "diagnosis": "glioma",
            "severity": "high",
            "case": "patient experienced progressive motor weakness and intracranial pressure symptoms. scan demonstrates infiltrative intra-axial glioma requiring prompt decompression.",
            "rec": "neurosurgical resection with intraoperative guidance and adjuvant radiation planning.",
        },
        {
            "finding": "deep parenchymal heterogeneous lesion causing midline shift and cerebral edema",
            "diagnosis": "glioma",
            "severity": "critical",
            "case": "subacute onset of confusion and contralateral hemiparesis. neuroimaging confirms aggressive astrocytic neoplasm.",
            "rec": "immediate corticosteroid edema management and emergent neurosurgical evaluation.",
        },
    ],
    "meningioma_tumor": [
        {
            "finding": "well-circumscribed extra-axial dural-based mass along convexity with dural tail sign",
            "diagnosis": "meningioma",
            "severity": "moderate",
            "case": "patient presented with persistent chronic headache and localized cranial pressure. scan confirms dural-attached extra-axial neoplasm without parenchymal invasion.",
            "rec": "neurosurgical evaluation for elective craniotomy resection and contrast mri follow-up.",
        },
        {
            "finding": "homogeneously enhancing dural-based mass compressing adjacent brain cortex",
            "diagnosis": "meningioma",
            "severity": "moderate",
            "case": "incidental and symptomatic findings of benign extra-axial convexity lesion causing focal mass effect.",
            "rec": "preoperative embolization assessment and complete surgical excision.",
        },
        {
            "finding": "parasagittal extra-axial mass with characteristic broad dural attachment",
            "diagnosis": "meningioma",
            "severity": "mild",
            "case": "gradual onset of dull cranial discomfort. benign extra-axial pathology typical of low-grade meningioma.",
            "rec": "regular interval neuroimaging surveillance or elective micro-surgical resection.",
        },
    ],
    "pituitary_tumor": [
        {
            "finding": "sellar and suprasellar mass expanding pituitary fossa with optic chiasm compression",
            "diagnosis": "pituitary adenoma",
            "severity": "moderate",
            "case": "patient presented with bitemporal visual field deficits and endocrine disruption. scan demonstrates macroadenoma abutting optic apparatus.",
            "rec": "endocrinology workup, visual field testing, and transsphenoidal surgical decompression.",
        },
        {
            "finding": "expanded sella turcica with homogeneous suprasellar mass extending toward chiasm",
            "diagnosis": "pituitary adenoma",
            "severity": "moderate",
            "case": "evaluation for galactorrhea and peripheral visual changes. imaging reveals pituitary macroadenoma.",
            "rec": "hormone level panel, ophthalmology perimetry, and transsphenoidal resection.",
        },
        {
            "finding": "intrasellar lesion with suprasellar bulge without cavernous sinus invasion",
            "diagnosis": "pituitary adenoma",
            "severity": "mild",
            "case": "pituitary adenoma causing pituitary axis disruption without acute visual compromise.",
            "rec": "dopamine agonist therapy trial if prolactinoma, or elective transsphenoidal surgery.",
        },
    ],
    "no_tumor": [
        {
            "finding": "normal brain parenchyma with symmetric ventricles and preserved cortical sulci",
            "diagnosis": "normal brain mri",
            "severity": "normal",
            "case": "patient evaluated for non-specific tension headache. scan shows no acute intracranial pathology, midline shift, hemorrhage, or mass lesion.",
            "rec": "routine outpatient management; no acute surgical or oncological intervention indicated.",
        },
        {
            "finding": "unremarkable cerebral and cerebellar architecture without mass or edema",
            "diagnosis": "normal brain mri",
            "severity": "normal",
            "case": "screening scan for dizziness. intact ventricular system and clean parenchymal signal intensity throughout.",
            "rec": "conservative symptomatic care; reassuring examination.",
        },
        {
            "finding": "symmetric brain parenchyma with normal grey-white differentiation and no enhancement",
            "diagnosis": "normal brain mri",
            "severity": "normal",
            "case": "reassurance scan following minor head impact. no focal contusion, subdural collection, or lesion.",
            "rec": "discharge to primary care follow-up as clinically indicated.",
        },
    ],
}


class BrainMRIVLMDataset(Dataset):
    """PyTorch Dataset loading Brain MRI scans + generating paired clinical case reports."""

    def __init__(
        self,
        root_dir: str = "data/brain_mri_raw",
        split: str = "Training",
        tokenizer: MedicalTokenizer = None,
        img_size: int = 128,
        max_seq_len: int = 160,
    ):
        self.root_dir = root_dir
        self.split = split
        self.tokenizer = tokenizer or MedicalTokenizer()
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, str]] = []

        categories = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]
        split_dir = os.path.join(root_dir, split)

        for cat in categories:
            cat_dir = os.path.join(split_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cat_dir, fname), cat))

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, category = self.samples[idx]

        # 1. Load & Preprocess MRI Image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # Normalize to [-1, 1] range standard
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 2. Select Prompt & Clinical Diagnostic Case
        prompt_tmpl = random.choice(PROMPT_TEMPLATES)
        case_item = random.choice(DIAGNOSTIC_CASES[category])

        response_dict = {
            "modality": "brain mri",
            "finding": case_item["finding"],
            "diagnosis": case_item["diagnosis"],
            "severity": case_item["severity"],
            "clinical_case": case_item["case"],
            "recommendations": case_item["rec"],
        }
        # Format compact JSON
        response_json = (
            f'{{"modality":"{response_dict["modality"]}",'
            f'"finding":"{response_dict["finding"]}",'
            f'"diagnosis":"{response_dict["diagnosis"]}",'
            f'"severity":"{response_dict["severity"]}",'
            f'"clinical_case":"{response_dict["clinical_case"]}",'
            f'"recommendations":"{response_dict["recommendations"]}"}}'
        )

        # 3. Tokenize Sequence: [BOS] Prompt [SEP] Target [EOS]
        prompt_ids = self.tokenizer.encode(prompt_tmpl, add_bos=True, add_eos=False)
        prompt_ids.append(self.tokenizer.sep_token_id)
        
        target_ids = self.tokenizer.encode(response_json, add_bos=False, add_eos=True)

        # Full sequence
        full_seq = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

        # Build input_ids and targets (causal shift)
        # input_ids: full_seq[:-1]
        # targets: full_seq[1:]
        # loss_mask: 0.0 on prompt tokens, 1.0 on target response tokens
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[:self.max_seq_len]

        seq_len = len(full_seq)
        input_ids = full_seq[:-1]
        targets = full_seq[1:]

        # Mask loss for prompt tokens (model should only be penalized for predicting target clinical response)
        loss_mask = [0.0] * (prompt_len - 1) + [1.0] * (seq_len - prompt_len)

        # Pad to max_seq_len - 1
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            targets = targets + [self.tokenizer.pad_token_id] * pad_len
            loss_mask = loss_mask + [0.0] * pad_len

        return {
            "images": img_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "category": category,
        }


def get_dataloaders(
    root_dir: str = "data/brain_mri_raw",
    batch_size: int = 32,
    img_size: int = 128,
    max_seq_len: int = 160,
) -> Tuple[DataLoader, DataLoader, MedicalTokenizer]:
    tok = MedicalTokenizer("models/vocab.json")
    train_ds = BrainMRIVLMDataset(root_dir, split="Training", tokenizer=tok, img_size=img_size, max_seq_len=max_seq_len)
    test_ds = BrainMRIVLMDataset(root_dir, split="Testing", tokenizer=tok, img_size=img_size, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, tok


if __name__ == "__main__":
    train_loader, val_loader, tok = get_dataloaders(batch_size=4)
    print(f"Train samples: {len(train_loader.dataset)}, Batches: {len(train_loader)}")
    print(f"Test samples: {len(val_loader.dataset)}, Batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    print("Batch images:", batch["images"].shape)
    print("Batch input_ids:", batch["input_ids"].shape)
    print("Batch targets:", batch["targets"].shape)
    print("Batch loss_mask sum:", batch["loss_mask"].sum().item())
