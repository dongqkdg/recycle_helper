"""Build-time check that dependencies and pretrained model weights are usable."""

from PIL import Image

from app import classify


image = Image.new("RGB", (224, 224), (128, 128, 128))
category, confidence, model = classify(image, "docker-smoke-test.jpg")
assert category in {"플라스틱", "종이", "캔", "유리", "일반쓰레기"}
assert 0.0 <= confidence <= 1.0
assert model in {"MobileNetV3-Small (ImageNet)", "EfficientNet-B0 (ImageNet)"}
print(f"Model smoke test passed: {model}, {category}, confidence={confidence}")
