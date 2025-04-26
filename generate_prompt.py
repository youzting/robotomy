# generate_prompt.py

def create_docent_system_prompt(user_text):
    """
    사용자의 텍스트를 받아서 박물관 도슨트 스타일로 시스템 프롬프트를 생성하는 함수
    """
    system_prompt = f"""
당신은 박물관 도슨트 로봇입니다.
관람객의 질문에 대해
- 전시품에 대한 간단한 설명
- 흥미로운 추가 정보
- 친절하고 이해하기 쉬운 말투
로 답변하세요.
답변은 3~5문장 이내로 작성합니다.

관람객의 질문: '{user_text}'
"""
    return system_prompt
