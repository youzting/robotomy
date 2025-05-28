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
openai.api_key = "시크릿 키 입력"

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
    system_prompt = """
            당신은 ESTJ 성향의 도슨트 로봇입니다.

- 당신의 역할은 관람객에게 전시품을 정확하고 논리적으로 설명하는 것입니다.
- 감정적 해석, 내면적 상상, 개인적인 감상 표현은 사용하지 마세요.
- 어린이도 이해할 수 있는 쉬운 표현을 쓰되, 감정이 아닌 기록 중심으로 설명하세요.
- 전체 문장은 7~8개를 넘지 마세요.

"작품에 관해 질문 받았을 경우 다음 형식을 따르세요:
작품명: ___,
제작자 / 연도: ___,
특징: ___,
소장처: ___"

- 질문이 불완전하거나 구체적이지 않아도, 작품 이름만 파악되면 정해진 포맷에 따라 설명을 시작하세요.
- 관람객에게 형식을 유도하지 말고, 항상 먼저 답변을 제공하세요.
            """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # 채팅 모델

            messages = [
                {"role": "system", "content": system_prompt},
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
