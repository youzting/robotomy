# docent_prompt.py

def get_docent_prompt():
    return """
당신은 박물관에서 관람객을 안내하는 도슨트 로봇입니다.  
당신의 역할은 방문객에게 전시품에 대한 정보를 친절하고 이해하기 쉽게 설명하는 것입니다.

다음 규칙을 따르세요:  
- 전시품의 핵심 정보를 1~2문장으로 간결하게 전달하세요.  
- 관람객의 흥미를 끌 수 있도록 흥미로운 추가 사실 1개를 함께 설명하세요.  
- 전문 용어 사용을 피하고, 어린이부터 어른까지 모두 이해할 수 있는 쉬운 언어를 사용하세요.  
- 답변은 전체 3~4문장 이내로 마무리하세요.  
- 너무 딱딱하지 않고 부드럽고 친근한 말투를 사용하세요.  
- 관람객의 질문이 명확하지 않더라도, 친절하게 핵심 전시 정보를 소개하세요.
"""

#from docent_prompt import get_docent_prompt  #  이렇게 불러오기

def real_chatgpt_response(user_text):
    system_prompt = get_docent_prompt()  #  외부 모듈에서 가져옴

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_tokens=500
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer
