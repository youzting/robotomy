// TTS 예제 (텍스트 -> 음성)
// 생성된 음성 파일은 예제 코드와 같은 폴더에 위치함
// 영어 모델은 현재 TTS 라이브러리에서 지원하지만, 한국어 모델은 지원하지 않고 있음
// 다국어 지원 모델을 연구하여 한국어가 출력되게 해야 함

from TTS.api import TTS

# 모델 로드 (최초 실행 시 다운로드됨)
# 사용 모델 (tacotron2)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=False)

# 텍스트 → 오디오
tts.tts_to_file(text="Hello, how are you today?", file_path="output.wav")

print("✅ output.wav 생성 완료")
