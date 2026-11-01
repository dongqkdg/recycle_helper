# 분리수거 쓰레기 분류 웹앱

Python과 FastAPI로 만든 로컬 분리수거 안내 앱입니다. 이미지를 업로드하면 미리보기와 함께 쓰레기 종류, 추정 신뢰도, 분리수거 방법을 보여줍니다.

## 설치

Python 3.10 이상을 권장합니다. 가상환경 없이도 실행할 수 있습니다.

```powershell
cd "upload 폴더의 경로"
python -m pip install -r requirements.txt
```

## 모델 준비

실제 이미지 분류에는 CPU용 PyTorch와 torchvision이 필요합니다.

```powershell
python -m pip install torch torchvision
```

앱은 첫 분류 시 `MobileNetV3-Small` ImageNet 사전학습 가중치를 사용합니다. 최초 실행 시 가중치 다운로드가 필요하며, 인터넷이 제한된 환경에서는 다운로드가 실패할 수 있습니다. 이 경우 앱은 임의의 분류를 하지 않고 모델 미설치 오류를 표시합니다.

## 실행

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

브라우저에서 http://127.0.0.1:8000 을 열고 이미지를 선택하세요. 종료하려면 실행 중인 PowerShell에서 `Ctrl+C`를 누릅니다.

## 주의사항

분류 결과는 AI의 추정치이며 실제 지자체 분리수거 기준과 다를 수 있습니다. 배출 전 거주 지역의 공식 규정을 확인하세요.
