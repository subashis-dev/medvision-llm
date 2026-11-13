import os
from inference import MedVisionPipeline

pipe = MedVisionPipeline()
samples = [
    ("GLIOMA", "data/samples/glioma_tumor/image(59).jpg"),
    ("MENINGIOMA", "data/samples/meningioma_tumor/image(10).jpg"),
    ("PITUITARY ADENOMA", "data/samples/pituitary_tumor/image(1).jpg"),
    ("NORMAL BRAIN MRI", "data/samples/no_tumor/image(1).jpg"),
]

print("\n" + "="*66)
print("  🏥 CLINICAL TEST GENERATION ACROSS 4 BRAIN MRI CATEGORIES")
print("="*66)

for cat, path in samples:
    if os.path.exists(path):
        res = pipe.analyze_scan(path)
        diag = res.get('diagnosis')
        conf = res.get('visual_confidence')
        status = "✅ MATCH" if cat.split()[0].lower() in diag.lower() else "❌ MISMATCH"
        print(f"Pathology Target: [{cat}] -> {status}")
        print(f"  -> Model Diagnosis: {diag} ({conf})")
        print(f"  -> Finding: {res.get('finding')}")
        print(f"  -> Severity: {res.get('severity')}")
        print(f"  -> Clinical Case: {res.get('clinical_case')[:80]}...\n")
print("="*66)
