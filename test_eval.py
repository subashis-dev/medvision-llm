import os
from inference import MedVisionPipeline

pipe = MedVisionPipeline()
samples = [
    ("GLIOMA", "data/samples/glioma_tumor/image(1).jpg"),
    ("MENINGIOMA", "data/samples/meningioma_tumor/image(1).jpg"),
    ("PITUITARY", "data/samples/pituitary_tumor/image(1).jpg"),
    ("NO TUMOR", "data/samples/no_tumor/image(1).jpg"),
]

print("\n" + "="*60)
print("  🏥 CLINICAL TEST GENERATION ACROSS 4 BRAIN MRI CATEGORIES")
print("="*60)

for cat, path in samples:
    if os.path.exists(path):
        res = pipe.analyze_scan(path)
        print(f"Pathology: [{cat}]")
        print(f"  -> Model Diagnosis: {res.get('diagnosis')}")
        print(f"  -> Finding: {res.get('finding')}")
        print(f"  -> Severity: {res.get('severity')}")
        print(f"  -> Clinical Case: {res.get('clinical_case')}\n")
print("="*60)
