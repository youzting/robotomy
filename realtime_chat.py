// 노트북 수리 완료 예정일 25.04.21.(월) 이후 코드 테스트 예정
// Whisper,ChatGPT, TTS 를 기반으로 한 실시간 대화 시스템의 코드를 설계해 둠

import whisper
from TTS.api import TTS
import os
import sounddevice as sd
import soundfile as sf

# --- STEP 1: Whisper 설정 ---
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("base")  # small, medium, large 선택 가능

# --- STEP 2: TTS 설정 ---
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# --- STEP 3: ChatGPT 응답 (예시) ---
def mock_chatgpt_response(user_text):
    # 실제 ChatGPT API 연결 대신 간단히 응답 예시
    return f"당신이 이렇게 말했어요: '{user_text}' 좋은 하루 되세요!"

# --- STEP 4: 음성 재생 함수 ---
def play_audio(file_path):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

# --- STEP 5: 대화 루프 ---
def main():
    # 예시용 음성 파일 입력 (Whisper용)
    input_audio = "user_input.wav"  # 실제 사용 시 mic 녹음으로 확장 가능
    speaker_audio = "audio_test_ko.wav"  # 화자 음성 예시 (한국어 TTS 스타일링용)

    # 1. Whisper로 음성 → 텍스트
    print("🎙️ 음성 → 텍스트 처리 중...")
    result = whisper_model.transcribe(input_audio)
    user_text = result["text"]
    print("📝 사용자의 발화:", user_text)

    # 2. ChatGPT 응답 생성
    response_text = mock_chatgpt_response(user_text)
    print("🤖 ChatGPT 응답:", response_text)

    # 3. TTS로 응답 음성 생성
    output_audio = "response.wav"
    tts.tts_to_file(
        text=response_text,
        file_path=output_audio,
        speaker_wav=speaker_audio,
        language="ko"
    )
    print("🔊 응답 음성 생성 완료:", output_audio)

    # 4. 음성 재생
    play_audio(output_audio)

if __name__ == "__main__":
    main()
