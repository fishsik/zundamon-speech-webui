import os
import torch
import numpy as np
from scipy.signal import butter, lfilter
from f5_tts.api import F5TTS
import soundfile as sf

class AIVTuberTTS:
    def __init__(self, ref_audio_path="reference/reference.wav", ref_text_path="reference/ref_text.txt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing F5-TTS on {self.device}...")
        
        # 모델을 한 번만 로드하여 메모리에 유지 (Singleton 패턴)
        # F5-TTS Base 모델은 경량화되어 있으면서도 제로샷 성능이 뛰어남
        self.f5_tts = F5TTS(model="F5TTS_Base", device=self.device)
        
        self.ref_audio_path = ref_audio_path
        self.ref_text = ""
        
        if os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                self.ref_text = f.read().strip()
                
        os.makedirs("output", exist_ok=True)

    def _apply_audio_processing(self, wav_path):
        """ 오래된 영상 같은 칙칙한 느낌을 줄이고 깔끔하게 다듬는 후처리 필터 """
        try:
            data, samplerate = sf.read(wav_path)
            
            # 1. 미세한 고주파 잡음(Hiss)을 부드럽게 걸러주는 간단한 로우패스/하이패스 필터 적용 예시
            # 여기서는 음질을 선명하게 만들기 위해 데이터 정규화 및 DC offset 제거 수행
            data = data - np.mean(data) # 중심 맞추기
            data = data / (np.max(np.abs(data)) + 1e-8) * 0.95 # 피크 노멀라이징 (클리핑 방지)
            
            sf.write(wav_path, data, samplerate)
        except Exception as e:
            print(f"오디오 후처리 중 예외 발생 (무시 가능): {e}")

    def generate_speech(self, text: str, output_filename="response.wav"):
        output_path = os.path.join("output", output_filename)
        
        try:
            # F5-TTS를 통한 음성 합성 (일본어 레퍼런스 기반 영어/다국어 발화 지원)
            self.f5_tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=text,
                file_wave=output_path,
                speed=0.8,
                nfe_step=32
            )

            self._apply_audio_processing(output_path)
            
            return output_path
        except Exception as e:
            print(f"TTS 생성 중 에러 발생: {e}")
            return None

# 사용 예시 (VTuber 에이전트 루프 연동 시)
if __name__ == "__main__":
    tts_module = AIVTuberTTS()
    
    # LLM이 생성한 영어 응답 가정
    english_response = "Hello! I am your AI VTuber, streaming live with a cloned voice."
    
    result_path = tts_module.generate_speech(english_response)
    if result_path:
        print(print(f"음성 파일 생성 완료: {result_path}"))