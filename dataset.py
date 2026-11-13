import os
import random
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import MedicalTokenizer

# 12 Diverse Clinical Question & Diagnostic Task Prompts
CAT2LABEL = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor": 2,
    "no_tumor": 3,
}

PROMPT_TEMPLATES = [
    "Scan: Brain MRI | Task: Analyze clinical case and diagnose",
    "Scan: Brain MRI | Query: What pathology is visible in this scan?",
    "Scan: Brain MRI | Question: Explain the clinical case and what happened",
    "Scan: Brain MRI | Clinical Request: Provide diagnosis, severity, and recommendations",
    "Scan: Brain MRI | Task: Complete neuro-radiology evaluation",
    "Scan: Brain MRI | Triage: Is there an acute surgical lesion or tumor present?",
    "Scan: Brain MRI | Pathology Query: Identify tumor type and tissue origin",
    "Scan: Brain MRI | Anatomical Assessment: Evaluate mass effect and ventricle compression",
    "Scan: Brain MRI | Management: What are the immediate clinical recommendations?",
    "Scan: Brain MRI | Diagnostic Summary: State primary diagnosis and severity",
    "Scan: Brain MRI | Case History: Correlate imaging findings with patient symptoms",
    "Scan: Brain MRI | Radiologist Report: Full structured neuroimaging findings",
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
        {
            "finding": "irregular ring-enhancing intra-axial lesion with central necrotic changes and surrounding flare edema",
            "diagnosis": "glioma",
            "severity": "critical",
            "case": "elderly patient presented with rapid cognitive decline and nocturnal headaches. scan shows glioblastoma-type feature with significant mass displacement.",
            "rec": "maximal safe surgical debulking followed by concurrent chemoradiotherapy protocol.",
        },
        {
            "finding": "hyperintense frontal lobe mass demonstrating expansile growth without clear margins",
            "diagnosis": "glioma",
            "severity": "high",
            "case": "adult presenting with personality changes and refractory seizures. intra-axial astrocytic tumor identified within anterior cranial compartment.",
            "rec": "awake functional craniotomy resection and molecular histopathological subtyping.",
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
        {
            "finding": "sphenoid wing dural mass with bony hyperostosis and mild proptosis",
            "diagnosis": "meningioma",
            "severity": "moderate",
            "case": "patient noted progressive visual blurring and temporal fullness. imaging demonstrates skull-base meningioma with vascular abutment.",
            "rec": "skull-base neurosurgical consultation with visual field monitoring and stereotactic radiosurgery consideration.",
        },
        {
            "finding": "falx cerebri extra-axial lesion showing dense homogeneous contrast uptake",
            "diagnosis": "meningioma",
            "severity": "moderate",
            "case": "intermittent headaches and localized tenderness. imaging displays classic non-invasive dural-based meningioma.",
            "rec": "Simpson grade I/II surgical resection and long-term annual surveillance MRI.",
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
        {
            "finding": "large sellar mass causing optic chiasmatic elevation and pituitary stalk deviation",
            "diagnosis": "pituitary adenoma",
            "severity": "high",
            "case": "patient presented with acute onset bilateral hemianopsia and hypopituitarism symptoms. imaging shows non-functioning macroadenoma.",
            "rec": "urgent transsphenoidal endoscopic resection to salvage optic nerve function.",
        },
        {
            "finding": "focal adenohypophyseal adenoma confined to sella with mild floor remodeling",
            "diagnosis": "pituitary adenoma",
            "severity": "mild",
            "case": "endocrine screening for hypercortisolemia. scan identifies microadenoma consistent with functional pituitary pathology.",
            "rec": "endocrinologist consultation, dynamic hormonal suppression testing, and targeted medical management.",
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
        {
            "finding": "age-appropriate cerebral volume without focal acute ischemia, mass lesion, or abnormal enhancement",
            "diagnosis": "normal brain mri",
            "severity": "normal",
            "case": "investigation for chronic migraine with aura. scan confirms clear basal cisterns and absence of neoplasm.",
            "rec": "neurology headache clinic optimization with lifestyle and prophylactic therapy.",
        },
        {
            "finding": "intact craniocervical junction, patent dural sinuses, and normal cerebral hemispheres",
            "diagnosis": "normal brain mri",
            "severity": "normal",
            "case": "diagnostic workup for transient paresthesia. scan confirms completely normal neuroaxis with zero intracranial mass.",
            "rec": "reassurance provided; no specialized neurosurgical follow-up necessary.",
        },
    ],
}


def apply_medical_augmentation(img: np.ndarray) -> np.ndarray:
    """Applies realistic dynamic medical imaging augmentations (rotation, zoom, contrast, blur)."""
    h, w = img.shape[:2]

    # 1. Random Horizontal Flip (Brain bilateral symmetry)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # 2. Random Affine Rotation (+/- 14 degrees) and Scaling (0.92 - 1.08x)
    angle = random.uniform(-14.0, 14.0)
    scale = random.uniform(0.92, 1.08)
    tx = random.uniform(-6.0, 6.0)
    ty = random.uniform(-6.0, 6.0)

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3. Dynamic Contrast & Gamma Adjustments (Simulating 1.5T vs 3.0T scanner variations)
    gamma = random.uniform(0.85, 1.20)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 4. Subtle Gaussian noise / smoothing
    if random.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


class BrainMRIVLMDataset(Dataset):
    """Multi-source Medical Vision-Language Dataset with real-time online data augmentation."""

    def __init__(
        self,
        split: str = "Training",
        tokenizer: MedicalTokenizer = None,
        img_size: int = 128,
        max_seq_len: int = 160,
        augment: bool = True,
    ):
        self.split = split
        self.tokenizer = tokenizer or MedicalTokenizer()
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.augment = augment and (split == "Training")
        self.samples: List[Tuple[str, str]] = []

        cat_map = {
            "glioma_tumor": "glioma_tumor",
            "meningioma_tumor": "meningioma_tumor",
            "pituitary_tumor": "pituitary_tumor",
            "no_tumor": "no_tumor",
            "glioma": "glioma_tumor",
            "meningioma": "meningioma_tumor",
            "pituitary": "pituitary_tumor",
            "notumor": "no_tumor",
        }

        # 1. Source A: data/brain_mri_raw
        src_a = os.path.join("data", "brain_mri_raw", split)
        if os.path.exists(src_a):
            for folder in os.listdir(src_a):
                std_cat = cat_map.get(folder.lower())
                if not std_cat:
                    continue
                cat_path = os.path.join(src_a, folder)
                for fn in os.listdir(cat_path):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cat_path, fn), std_cat))

        # 2. Source B: data/neuroscan_raw/brain_data (Secondary large clinical dataset)
        src_b = os.path.join("data", "neuroscan_raw", "brain_data", split)
        if os.path.exists(src_b):
            for folder in os.listdir(src_b):
                std_cat = cat_map.get(folder.lower())
                if not std_cat:
                    continue
                cat_path = os.path.join(src_b, folder)
                for fn in os.listdir(cat_path):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cat_path, fn), std_cat))

        # Fallback to data/samples if raw datasets not present
        if not self.samples:
            samples_dir = os.path.join("data", "samples")
            if os.path.exists(samples_dir):
                for cat in ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]:
                    cp = os.path.join(samples_dir, cat)
                    if os.path.exists(cp):
                        for fn in os.listdir(cp):
                            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                                self.samples.append((os.path.join(cp, fn), cat))

        # Deterministic shuffle
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, category = self.samples[idx]

        # 1. Load & Preprocess MRI Image with Out-of-Memory Safeguard
        try:
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 2. Apply Dynamic Medical Augmentation on Training Split
        if self.augment:
            img = apply_medical_augmentation(img)

        # 3. Normalize to ImageNet Distribution
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 4. Select from 12 Clinical Prompts & Diverse Case Variations
        prompt_tmpl = random.choice(PROMPT_TEMPLATES)
        case_item = random.choice(DIAGNOSTIC_CASES[category])

        response_dict = {
            "diagnosis": case_item["diagnosis"],
            "severity": case_item["severity"],
            "finding": case_item["finding"],
            "modality": "brain mri",
            "clinical_case": case_item["case"],
            "recommendations": case_item["rec"],
        }
        response_json = (
            f'{{"diagnosis":"{response_dict["diagnosis"]}",'
            f'"severity":"{response_dict["severity"]}",'
            f'"finding":"{response_dict["finding"]}",'
            f'"modality":"{response_dict["modality"]}",'
            f'"clinical_case":"{response_dict["clinical_case"]}",'
            f'"recommendations":"{response_dict["recommendations"]}"}}'
        )

        # 5. Tokenize: [BOS] Prompt [SEP] Target [EOS]
        prompt_ids = self.tokenizer.encode(prompt_tmpl, add_bos=True, add_eos=False)
        prompt_ids.append(self.tokenizer.sep_token_id)
        target_ids = self.tokenizer.encode(response_json, add_bos=False, add_eos=True)

        full_seq = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[: self.max_seq_len]

        seq_len = len(full_seq)
        input_ids = full_seq[:-1]
        targets = full_seq[1:]

        # Mask loss for prompt tokens
        loss_mask = [0.0] * (prompt_len - 1) + [1.0] * (seq_len - prompt_len)

        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            targets = targets + [self.tokenizer.pad_token_id] * pad_len
            loss_mask = loss_mask + [0.0] * pad_len

        cat_idx = CAT2LABEL.get(category, 0)
        return {
            "images": img_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "category": category,
            "labels": torch.tensor(cat_idx, dtype=torch.long),
        }


def get_dataloaders(
    batch_size: int = 32,
    img_size: int = 128,
    max_seq_len: int = 160,
) -> Tuple[DataLoader, DataLoader, MedicalTokenizer]:
    tok = MedicalTokenizer("data/vocab.json")
    train_ds = BrainMRIVLMDataset(split="Training", tokenizer=tok, img_size=img_size, max_seq_len=max_seq_len, augment=True)
    val_ds = BrainMRIVLMDataset(split="Testing", tokenizer=tok, img_size=img_size, max_seq_len=max_seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, tok


if __name__ == "__main__":
    train_loader, val_loader, tok = get_dataloaders(batch_size=8)
    print(f"\n🚀 MULTI-SOURCE MERGED DATASET READY!")
    print(f"  Training Scans : {len(train_loader.dataset):,} real MRI scans")
    print(f"  Testing Scans  : {len(val_loader.dataset):,} real MRI scans")
    print(f"  Total Dataset  : {len(train_loader.dataset) + len(val_loader.dataset):,} unique Brain MRI scans!")
