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
    UNKNOWN = 'unknown'


_STOP_KEYWORDS = ('멈춰', '정지')
_RESET_KEYWORDS = ('리셋', '초기화')

# MANUAL 이동 명령: 발화에 포함된 구문 -> named_target
_MANUAL_MOVE_KEYWORDS = {
    '홈으로 가': 'home',
    '정면을 봐': 'front',
    '위를 봐': 'up',
    '아래를 봐': 'down',
    '컨베이어를 봐': 'watch',
}

_MODE_MANUAL_KEYWORDS = ('수동',)
_MODE_AUTO_KEYWORDS = ('자동',)

# AUTO 공구 전달 명령: 발화에 포함된 공구 이름 -> tool_class
_TOOL_KEYWORDS = {
    '스패너': 'spanner',
    '드라이버': 'driver',
    '렌치': 'wrench',
    '펜치': 'pliers',
    '망치': 'hammer',
    '물병': 'water_bottle',
}


def parse_command(text: str) -> dict:
    """발화를 해석해 명령 dict를 반환한다.

    반환 형태:
      {'type': Command.STOP}
      {'type': Command.RESET}
      {'type': Command.MOVE_NAMED, 'named_target': 'home'}
      {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}
      {'type': Command.FETCH_TOOL, 'tool': 'spanner'}
      {'type': Command.UNKNOWN, 'text': text}
    """
    stripped = text.strip()

    if any(keyword in stripped for keyword in _STOP_KEYWORDS):
        return {'type': Command.STOP}

    if any(keyword in stripped for keyword in _RESET_KEYWORDS):
        return {'type': Command.RESET}

    for phrase, named_target in _MANUAL_MOVE_KEYWORDS.items():
        if phrase in stripped:
            return {'type': Command.MOVE_NAMED, 'named_target': named_target}

    if any(keyword in stripped for keyword in _MODE_MANUAL_KEYWORDS):
        return {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}
    if any(keyword in stripped for keyword in _MODE_AUTO_KEYWORDS):
        return {'type': Command.MODE_SWITCH, 'mode': Mode.AUTO}

    for keyword, tool in _TOOL_KEYWORDS.items():
        if keyword in stripped:
            return {'type': Command.FETCH_TOOL, 'tool': tool}

    return {'type': Command.UNKNOWN, 'text': stripped}
