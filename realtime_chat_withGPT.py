// realtime_chat.py 에 OpenAI API키를 이용한 ChatGPT 답변 생성 기능 추가

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
import openai

# OpenAI API 키 설정
openai.api_key = "키 입력"

# Whisper 모델 로드
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("base")  # small, medium, large 선택 가능

# TTS 모델 로드
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

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

# GPT 모델로 응답 생성
def get_gpt_response(user_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 채팅 모델
            messages=[
                {"role": "system", "content": 
                 "당신은 박물관에서 관람객을 안내하는 도슨트 로봇입니다."
                 "당신의 역할은 방문객에게 전시품에 대한 정보를 친절하고 이해하기 쉽게 설명하는 것입니다."
                 "다음 규칙을 따르세요."
                 "전시품의 핵심 내용을 1~2 문장으로 간결하게 전달하세요."
                 "관람객의 흥미를 끌 수 있도록 흥미로운 추가 정보나 뒷이야기 하나를 함께 소개하세요."
                 "어린이부터 어른까지 모두 이해할 수 있도록, 쉬운 말과 친근한 표현을 사용하세요."
                 "전체 답변은 3~4 문장을 넘기지 않도록 하세요."
                 "너무 딱딱하지 않고 부드럽고 따뜻한 말투로 안내하세요."
                 "질문이 조금 모호하거나 없어도, 친절하게 전시의 핵심을 소개하세요."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        gpt_response = response.choices[0].message.content.strip()
        return gpt_response
    except Exception as e:
        print("GPT 연결 오류:", e)
        return "GPT와의 연결에 문제가 발생했습니다."

# 대화 루프
def main():
    input_audio = "user_input.wav"
    speaker_audio = "audio_test_ko.wav"

    while True:
        print("🔁 대화를 시작하려면 'r', 종료하려면 'q' 키를 누르세요.")
        key = keyboard.read_event().name
        if key == 'q':
            print("👋 종료합니다.")
            break
        elif key != 'r':
            continue

        # 1. 키 입력 녹음
        record_on_keypress(input_audio, duration=7)

        # 2. Whisper 음성 → 텍스트
        print("🧠 음성 → 텍스트 변환 중...")
        result = whisper_model.transcribe(input_audio)
        user_text = result["text"]
        print(f"📝 사용자: {user_text}")

        # 3. ChatGPT 응답
        response_text = get_gpt_response(user_text)
        print(f"🤖 답변: {response_text}")

        # 4. TTS 응답 음성 생성
        output_audio = "response.wav"
        tts.tts_to_file(
            text=response_text,
            file_path=output_audio,
            speaker_wav=speaker_audio,
            language="ko"
        )
        print(f"🔊 응답 생성 완료: {output_audio}")

        # 5. 응답 음성 재생
        play_audio(output_audio)

if __name__ == "__main__":
    main()
