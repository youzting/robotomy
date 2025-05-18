// Whisper,ChatGPT, TTS 를 기반으로 한 실시간 대화 시스템 ver.2.4
// 마이크 음성 입력 -> Whisper로 음성을 텍스트로 변환 -> GPT 응답 생성 -> 응답(텍스트)을 TTS로 변환 -> 음성 출력
// 사용자 발화 : user_input.wav 생성
// TTS 음성 : response.wav 생성
// 'r'키 입력 시 음성 녹음 시작
// 종료(Ctrl + C) 입력 전까지 대화 기능 반복

import whisper
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
add_safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})
from TTS.api import TTS
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import time
import keyboard

# Whisper 모델 로드
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("base")

# TTS 모델 로드
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# 대답 생성
def mock_chatgpt_response(user_text):
    return f"당신이 이렇게 말했어요: '{user_text}'"

# 녹음 함수 (지정 시간 녹음)
def record_on_keypress(filename, duration=5):
    print("⌨️  'r' 키를 누르면 녹음이 시작됩니다.")
    keyboard.wait('r')
    print("🎤 녹음 중... 말하세요.")
    recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, recording, 16000)
    print("✅ 녹음 완료.")

# 음성 재생
def play_audio(file_path):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

def main():
    input_audio = "user_input.wav"
    speaker_audio = "audio_test_ko.wav"

    while True:
        # 1. 키 입력 대기 후 녹음
        record_on_keypress(input_audio)

        # 2. Whisper 변환
        print("🧠 음성 → 텍스트 변환 중...")
        result = whisper_model.transcribe(input_audio)
        user_text = result["text"]
        print(f"📝 사용자: {user_text}")

        # 3. ChatGPT 응답
        response_text = mock_chatgpt_response(user_text)
        print(f"🤖 답변: {response_text}")

        # 4. TTS 변환
        output_audio = "response.wav"
        tts.tts_to_file(
            text=response_text,
            file_path=output_audio,
            speaker_wav=speaker_audio,
            language="ko"
        )
        print(f"🔊 응답 생성 완료: {output_audio}")

        # 5. 응답 재생
        play_audio(output_audio)

if __name__ == "__main__":
    main()
