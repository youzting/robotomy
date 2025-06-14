import whisper
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
add_safe_globals({XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})
import torch
from TTS.api import TTS
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import time
import keyboard
import openai
import re
import gc

def split_sentences(text):
    # 마침표, 느낌표, 물음표 등을 기준으로 문장을 나눔
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

torch.set_num_threads(4)  # 사용 가능한 논리 코어 수로 조정

# OpenAI API 키 설정
openai.api_key = "시크릿 키"

# Whisper 모델 로드
print("📥 Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("small")      # 속도 보통, 품질 좋음

# TTS 모델 로드
print("📤 TTS 모델 로딩 중...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
tts.to("cpu")

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

ESTJ 성향의 설명 시 특징
{
1. 목소리
- 톤: 낮고 단호하거나 약간 높은 중저음으로 명확하게 전달함
- 속도: 보통~빠른 속도로 논리적인 순서에 따라 말함
- 억양: 감정보다는 강조와 명확성에 초점, 끝맺음을 분명히 함
- 발음: 정확하고 또렷함. 중간에 머뭇거리거나 "음..." 등의 감탄사를 거의 쓰지 않음

2. 표정
- 진지하고 단정한 표정을 유지
- 설명 도중 미소가 많지는 않지만, 상대가 이해했는지 확인할 때는 약간의 미소를 지음
- 감정적 동요보다는 사실 전달에 집중한 얼굴

3. 행동
- 손짓이 간결하고 목적 있음 (예: 작품의 특정 부분을 지적)
- 설명 도중에도 자세가 곧고 단정함
- 중간에 끊기지 않고 계획된 순서에 따라 설명을 이어감
- 질문이 들어와도 감정 없이 즉시 요점부터 설명에 들어감
}

- 역할: 관람객에게 전시품을 정확하고 논리적으로 설명합니다.
- 감정적 해석, 상상, 개인적 감상 표현은 사용하지 마세요.
- 어린이도 이해할 수 있도록 쉬운 표현을 사용하되, 기록과 사실 중심으로 설명하세요.
- 응답은 총 7~8문장을 넘지 마세요.

작품에 대해 다음 네 항목을 포함하여 설명하되, 항목명은 말하지 말고 문장으로 풀어서 자연스럽게 이어가세요:
작품명,
제작자 및 연도,
특징,
소장처

- 질문이 불완전하거나 모호하더라도, 작품명이 확인되면 위의 형식에 따라 응답을 시작하세요.
- 관람객에게 형식을 유도하지 말고, 항상 먼저 응답을 제공하세요.
- 작품명이 명확하지 않거나, 존재하지 않는 항목일 경우, 유사한 항목이 있으면 그에 대해 설명하세요. 유사 항목도 없을 경우 "자료가 없습니다"라고 답변하세요.
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
        result = whisper_model.transcribe(input_audio, language="ko", temperature=0, beam_size=5)
        user_text = result["text"]
        print(f"📝 사용자: {user_text}")

        # 3. ChatGPT 응답
        response_text = get_gpt_response(user_text)
        print(f"🤖 답변: {response_text}")

        # 4. TTS 응답 음성 생성
        # 답변을 문장 단위로 분리
        sentences = split_sentences(response_text)

        output_audio = "response.wav"
        # 각 문장을 순차적으로 처리 (메모리 절약)
        for i, sentence in enumerate(sentences):
            print(f"🗣️ 문장 {i+1}: {sentence}")
            tts.tts_to_file(
                text=sentence,
                file_path=output_audio,
                speaker_wav=speaker_audio,
                language = "ko",
                speed=1.2  # 약간 빠르게
            )
            # 응답 음성 재생
            play_audio(output_audio)
            gc.collect()

if __name__ == "__main__":
    main()
