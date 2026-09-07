import os
import torch
from TTS.api import TTS

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# XTTS-v2 모델 로드 (다국어 및 크로스링구얼 클로닝 지원)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# reference 폴더 경로 설정
ref_audio_path = "reference/reference.wav"
ref_text_path = "reference/ref_text.txt"

# ref_text.txt 파일에서 일본어 스크립트 읽기
if os.path.exists(ref_text_path):
    with open(ref_text_path, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()
else:
    reference_text = ""
    print("경고: ref_text.txt를 찾지 못했습니다. 텍스트 없이 오디오 특징만으로 클로닝을 시도합니다.")

# 생성할 영어 문장
target_text = "Hello, this is a test to generate English speech using a Japanese voice reference."

# 출력 디렉토리 생성
os.makedirs("output", exist_ok=True)
output_path = "output/generated_english.wav"

# 음성 생성 및 저장
tts.tts_to_file(
    text=target_text,
    file_path=output_path,
    speaker_wav=ref_audio_path,
    language="en",          # 출력할 언어 (영어)
    gpt_cond_len=3,         # 조건부 오디오 길이 (초)
    temperature=0.7         # 생성 다양성 조절
)

print(f"영어 음성 파일 생성 완료: {output_path}")