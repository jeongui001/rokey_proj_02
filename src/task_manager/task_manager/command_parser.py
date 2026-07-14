"""LLM 없이 간단한 한국어 키워드 매칭으로 /user_command/text 발화를 해석한다."""


class Mode:
    AUTO = 'AUTO'
    MANUAL = 'MANUAL'


class Command:
    STOP = 'stop'
    MOVE_NAMED = 'move_named'
    MODE_SWITCH = 'mode_switch'
    FETCH_TOOL = 'fetch_tool'
    RESET = 'reset'
    RESUME = 'resume'
    UNKNOWN = 'unknown'


_STOP_KEYWORDS = ('멈춰', '정지', '스톱', '중지', '그만', '일시정지')
_RESET_KEYWORDS = ('리셋', '초기화', '복구')  # '복구'는 operator_gui 버튼 라벨("복구 요청")과도 일치
_RESUME_KEYWORDS = ('재개', '계속', '이어서')

# 짧은 동사 조합 대신 문구 단위로 나열 - _FETCH_INTENTS와 겹치는 동사만 쓰면 fetch_tool과 오매칭된다.
_MANUAL_MOVE_KEYWORDS = {
    '홈으로 가': 'home',
    '집으로 가': 'home',
    '정면을 봐': 'front',
    '정면 봐': 'front',
    '앞을 봐': 'front',
    '앞 봐': 'front',
    '위를 봐': 'up',
    '위 봐': 'up',
    '아래를 봐': 'down',
    '아래 봐': 'down',
    '밑을 봐': 'down',
    '컨베이어를 봐': 'watch',
    '컨베이어 봐': 'watch',
    '벨트를 봐': 'watch',
}

_MODE_MANUAL_KEYWORDS = ('수동',)
_MODE_AUTO_KEYWORDS = ('자동',)
# 이 중 하나는 있어야 모드 전환으로 인식 - "자동차"처럼 의도 없는 문장의 오매칭을 막는다.
_MODE_INTENTS = ('모드', '전환', '변환', '으로 해', '로 해', '로 바꿔', '으로 바꿔')
_FETCH_INTENTS = ('가져', '갖다', '전달', '줘', '주세요')
_NEGATION_KEYWORDS = ('가져오지 마', '갖다주지 마', '전달하지 마', '하지 마')

# tool_class 값은 YOLO 학습 클래스명과 정확히 일치해야 한다 - vision_node가 그대로 실어 보내고
# _check_trigger가 문자열을 그대로 비교한다.
_TOOL_KEYWORDS = {
    '드라이버': 'screwdriver',
    '십자드라이버': 'screwdriver',
    '일자드라이버': 'screwdriver',
    '렌치': 'wrench',
    '망치': 'hammer',
    '물병': 'bottle',
}

SUPPORTED_TOOL_CLASSES = tuple(dict.fromkeys(_TOOL_KEYWORDS.values()))


def parse_command(text: str) -> dict:
    stripped = text.strip()

    if any(keyword in stripped for keyword in _STOP_KEYWORDS):
        return {'type': Command.STOP}

    if any(keyword in stripped for keyword in _RESET_KEYWORDS):
        return {'type': Command.RESET}

    if any(keyword in stripped for keyword in _RESUME_KEYWORDS):
        return {'type': Command.RESUME}

    for phrase, named_target in _MANUAL_MOVE_KEYWORDS.items():
        if phrase in stripped:
            return {'type': Command.MOVE_NAMED, 'named_target': named_target}

    has_mode_intent = any(intent in stripped for intent in _MODE_INTENTS)
    if has_mode_intent and any(keyword in stripped for keyword in _MODE_MANUAL_KEYWORDS):
        return {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}
    if has_mode_intent and any(keyword in stripped for keyword in _MODE_AUTO_KEYWORDS):
        return {'type': Command.MODE_SWITCH, 'mode': Mode.AUTO}

    for keyword, tool in _TOOL_KEYWORDS.items():
        if (keyword in stripped
                and any(intent in stripped for intent in _FETCH_INTENTS)
                and not any(negative in stripped for negative in _NEGATION_KEYWORDS)):
            return {'type': Command.FETCH_TOOL, 'tool': tool}

    return {'type': Command.UNKNOWN, 'text': stripped}
