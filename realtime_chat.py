// Whisper,ChatGPT, TTS 를 기반으로 한 실시간 대화 시스템 ver.2.3
// 마이크 음성 입력 -> Whisper로 음성을 텍스트로 변환 -> GPT 응답 생성 -> 응답(텍스트)을 TTS로 변환 -> 음성 출력
// 사용자 발화 : user_input.wav 생성
// TTS 음성 : response.wav 생성
// 음성 감지 시 음성 녹음 시작
// silence_duration을 넘어가면 음성 감지 종료
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
import threading
import time

# Whisper 모델 로드
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("base")  # small, medium, large 선택 가능

# TTS 모델 로드
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# 대답 생성
def mock_chatgpt_response(user_text):
    # 실제 ChatGPT API 연결 대신 간단히 응답 예시
    return f"당신이 이렇게 말했어요: '{user_text}' 좋은 하루 되세요!"

# 녹음 설정
samplerate = 16000
channels = 1
threshold = 500  # 음성 감지 임계값
silence_duration = 1.0  # 무음 시간 (초 단위)

q = queue.Queue()

# 실시간 녹음 콜백
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_dynamic(filename):
    print("🎙️ 대기 중... (말하면 녹음 시작)")
    recording = []
    silence_counter = 0
    speaking = False

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback, dtype='int16'):
        while True:
            try:
                data = q.get(timeout=1)
                volume_norm = np.linalg.norm(data) * 10

                if volume_norm > threshold:
                    if not speaking:
                        print("🎤 음성 감지! 녹음 시작")
                        speaking = True
                    recording.append(data)
                    silence_counter = 0
                else:
                    if speaking:
                        silence_counter += data.shape[0] / samplerate
                        recording.append(data)
                        if silence_counter > silence_duration:
                            print("🛑 음성 종료 감지")
                            break
            except queue.Empty:
                continue

    recording = np.concatenate(recording, axis=0)
    sf.write(filename, recording, samplerate)

# 음성 재생
def play_audio(file_path):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

def main():
    input_audio = "user_input.wav"
    speaker_audio = "audio_test_ko.wav"

    while True:
        # 1. 사람 말할 때까지 대기 + 녹음
        record_dynamic(input_audio)

        # 2. Whisper 변환
        print("🧠 음성 → 텍스트 변환 중...")
        result = whisper_model.transcribe(input_audio)
        user_text = result["text"]
        print(f"📝 사용자: {user_text}")

        # 3. ChatGPT 답변
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

        # 5. 재생
        play_audio(output_audio)

if __name__ == "__main__":
    main()
