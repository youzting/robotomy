//realtime_chat.py 에 OpenAI API키를 이용한 ChatGPT 답변 생성 코드를 추가한 버전

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
import openai

# OpenAI API 키 설정
openai.api_key = "키 입력"

# Whisper 모델 로드
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("base")  # small, medium, large 선택 가능

# TTS 모델 로드
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# 녹음 설정
samplerate = 16000
channels = 1
threshold = 1000  # 음성 감지 임계값
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

# GPT 모델로 응답 생성
def get_gpt_response(user_text):
    try:
        response = openai.completions.create(  # 새로운 방식으로 API 호출
            model="gpt-3.5-turbo",  # gpt-3.5-turbo 사용
            prompt=f"당신은 친절한 대화형 AI입니다. 사용자: {user_text}",
            max_tokens=150,
            temperature=0.7
        )
        gpt_response = response['choices'][0]['text'].strip()  # 새로운 응답 구조에 맞게 수정
        return gpt_response
    except Exception as e:
        print("GPT 연결 오류:", e)
        return "GPT와의 연결에 문제가 발생했습니다."

# 대화 루프
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
        response_text = get_gpt_response(user_text)
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
