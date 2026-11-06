"""Evaluate the old and candidate pretrained models on a fixed HF 100-image split."""

import json
import io
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import (EfficientNet_B0_Weights, MobileNet_V3_Small_Weights,
                                efficientnet_b0, mobilenet_v3_small)

SEED = 20261106
OUT = Path("model-evaluation")
MAP = {"cardboard": "종이", "paper": "종이", "glass": "유리", "metal": "캔", "plastic": "플라스틱", "trash": "일반쓰레기"}


def main():
    OUT.mkdir(exist_ok=True)
    archive = OUT / "trashnet-resized.zip"
    if not archive.exists():
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive), "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"], check=True)
    with zipfile.ZipFile(archive) as bundle:
        names = sorted({path.split("/")[1] for path in bundle.namelist() if path.count("/") >= 2 and path.split("/")[1] in MAP})
        files = {name: sorted(path for path in bundle.namelist() if path.startswith(f"dataset-resized/{name}/") and path.lower().endswith((".jpg", ".jpeg", ".png")))[:20] for name in names}
        selected = []
        for name in names:
            for path in files[name]:
                selected.append((Image.open(io.BytesIO(bundle.read(path))).convert("RGB"), MAP[name], name))
    selected = selected[:100]
    assert len(selected) == 100, len(selected)
    tune, evaluation = selected[:30], selected[30:]

    candidates = {
        "mobilenet_v3_small": (mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval(), MobileNet_V3_Small_Weights.DEFAULT),
        "efficientnet_b0": (efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval(), EfficientNet_B0_Weights.DEFAULT),
    }
    results = {}
    with torch.inference_mode():
        for name, (model, weights) in candidates.items():
            predictions = []
            for image, _, _ in selected:
                scores = model(weights.transforms()(image).unsqueeze(0))[0].softmax(0)
                predictions.append(int(scores.argmax()))
            # ImageNet labels are not TrashNet labels; report the app's mapped output.
            labels = weights.meta["categories"]
            mapped = []
            for index in predictions:
                text = labels[index].lower()
                mapped.append(next((category for category, words in {"플라스틱": ["bottle", "plastic", "container", "cup"], "종이": ["paper", "book", "envelope", "carton"], "캔": ["can", "tin"], "유리": ["glass", "jar"], "일반쓰레기": []}.items() if any(word in text for word in words)), "일반쓰레기"))
            truth = [item[1] for item in selected]
            results[name] = {"tuning": {"accuracy": accuracy_score(truth[:30], mapped[:30]), "macro_f1": f1_score(truth[:30], mapped[:30], average="macro", zero_division=0)}, "evaluation": {"accuracy": accuracy_score(truth[30:], mapped[30:]), "macro_f1": f1_score(truth[30:], mapped[30:], average="macro", zero_division=0)}}
    chosen = max(results, key=lambda name: (results[name]["tuning"]["accuracy"], results[name]["tuning"]["macro_f1"]))
    report = {"dataset": "garythung/trashnet", "license": "MIT", "seed": SEED, "total": 100, "tuning": 30, "evaluation": 70, "candidates": results, "selected_model": chosen, "attempts": 1}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.md").write_text(f"# 모델 업데이트 평가\n\n- 데이터셋: `garythung/trashnet` (MIT)\n- 고정 분할: 튜닝 30건 / 최종 평가 70건\n- 선택 모델: `{chosen}`\n\n| 모델 | 튜닝 정확도 | 최종 정확도 | 튜닝 Macro-F1 | 최종 Macro-F1 |\n|---|---:|---:|---:|---:|\n" + "\n".join(f"| {name} | {value['tuning']['accuracy']:.3f} | {value['evaluation']['accuracy']:.3f} | {value['tuning']['macro_f1']:.3f} | {value['evaluation']['macro_f1']:.3f} |" for name, value in results.items()), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
