import json
import os
import re
from typing import Dict, List, Union

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<sep>", "<unk>",
]
# 16 visual spatial tokens output by ConvNeXt stem
VISUAL_TOKENS = [f"<vis_{i}>" for i in range(16)]

CLINICAL_KEYWORDS = [
    # Modality & anatomy
    "brain", "mri", "scan", "axial", "sagittal", "coronal", "t1", "t2", "flair",
    "cerebral", "hemisphere", "frontal", "temporal", "parietal", "occipital",
    "ventricle", "sulci", "gyri", "dura", "sella", "chiasm", "optic",
    "parenchyma", "cortex", "midline", "posterior", "fossa",
    
    # Pathologies & findings
    "glioma", "meningioma", "pituitary", "adenoma", "normal", "tumor", "neoplasm",
    "lesion", "mass", "edema", "vasogenic", "heterogeneous", "homogeneous",
    "hyperintense", "hypointense", "enhancement", "circumscribed", "infiltrative",
    "dural", "tail", "calcification", "necrosis", "hemorrhage", "herniation",
    "shift", "compression", "intracranial", "pressure", "hydrocephalus",
    
    # Severity & presentation
    "severity", "critical", "high", "moderate", "mild", "benign", "acute",
    "chronic", "headache", "seizure", "deficit", "vision", "nausea",
    "symptom", "asymptomatic", "neurological", "pressure",
    
    # Recommendations & clinical actions
    "recommendation", "neurosurgical", "consultation", "resection", "craniotomy",
    "biopsy", "stereotactic", "radiation", "chemotherapy", "follow-up",
    "contrast", "monitoring", "oncology", "endocrine", "hormone",
    
    # JSON & schema keys
    "modality", "finding", "diagnosis", "clinical_case", "recommendations",
    "action", "summary", "patient", "explanation", "status", "abnormal", "true", "false",
]


class MedicalTokenizer:
    """Specialized Clinical Medical Tokenizer with visual token placeholders & clinical lexicon."""

    def __init__(self, vocab_path: str = "models/vocab.json"):
        self.vocab_path = vocab_path
        self.special_tokens = SPECIAL_TOKENS
        self.visual_tokens = VISUAL_TOKENS
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3
        self.unk_token_id = 4
        
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        if os.path.exists(vocab_path):
            self.load(vocab_path)
        else:
            self._build_vocab()

    def _build_vocab(self) -> None:
        vocab = list(self.special_tokens) + list(self.visual_tokens)
        
        # Add clinical keywords
        for kw in CLINICAL_KEYWORDS:
            if kw not in vocab:
                vocab.append(kw)
        
        # Add JSON syntax & punctuation tokens
        punct = list('{}[]":,.-_/?!+*()%=><;\'')
        for p in punct:
            if p not in vocab:
                vocab.append(p)
                
        # Printable ASCII characters (digits & letters)
        for i in range(32, 127):
            ch = chr(i)
            if ch not in vocab:
                vocab.append(ch)

        # Pad to multiple of 64 for optimal GPU tensor core alignment
        pad_target = ((len(vocab) + 63) // 64) * 64
        for idx in range(len(vocab), pad_target):
            vocab.append(f"<pad_{idx}>")

        self.token2id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id2token = {idx: tok for idx, tok in enumerate(vocab)}

    def save(self, path: str = "models/vocab.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def tokenize(self, text: str) -> List[str]:
        # Clean & split into words, special tokens, or punctuation characters
        tokens = []
        raw_tokens = re.findall(r'<[a-zA-Z0-9_]+>|[a-zA-Z0-9_]+|[^\s\w]', text)
        for t in raw_tokens:
            t_low = t.lower()
            if t in self.token2id:
                tokens.append(t)
            elif t_low in self.token2id:
                tokens.append(t_low)
            else:
                # Character fallback
                for ch in t:
                    tokens.append(ch if ch in self.token2id else "<unk>")
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for t in tokens:
            ids.append(self.token2id.get(t, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            if tid == self.eos_token_id and skip_special:
                break
            tok = self.id2token.get(tid, "<unk>")
            if skip_special and (tok in self.special_tokens or tok.startswith("<pad_") or tok.startswith("<vis_")):
                continue
            tokens.append(tok)
        
        # Detokenize nicely
        text = ""
        for i, tok in enumerate(tokens):
            if tok in '{}[],:":' or (i > 0 and tokens[i-1] in '{"[:'):
                text += tok
            elif len(tok) == 1 and i > 0 and len(tokens[i-1]) == 1 and tok.isalnum() and tokens[i-1].isalnum():
                text += tok
            else:
                text += (" " if text and not text.endswith(("{", "[", ":", " ", "\n", '"')) else "") + tok
        return text


if __name__ == "__main__":
    tok = MedicalTokenizer()
    tok.save("models/vocab.json")
    print(f"MedicalTokenizer built! Vocab size: {tok.vocab_size} (aligned to 64)")
    sample = '{"modality": "Brain MRI", "diagnosis": "Glioma", "severity": "High"}'
    encoded = tok.encode(sample, add_bos=True, add_eos=True)
    decoded = tok.decode(encoded)
    print("Sample encode len:", len(encoded))
    print("Sample decode:", decoded)
