from task_manager.command_parser import Command, Mode, parse_command


def test_stop_keyword():
    assert parse_command('멈춰') == {'type': Command.STOP}
    assert parse_command('로봇아 정지해줘') == {'type': Command.STOP}


def test_stop_keyword_synonyms():
    assert parse_command('스톱') == {'type': Command.STOP}
    assert parse_command('중지해줘') == {'type': Command.STOP}
    assert parse_command('그만') == {'type': Command.STOP}


def test_reset_keyword():
    assert parse_command('리셋') == {'type': Command.RESET}
    assert parse_command('초기화 해줘') == {'type': Command.RESET}


def test_reset_keyword_synonym_matches_gui_button_label():
    # operator_gui의 복구 버튼 라벨이 "복구 요청 (리셋)"이라 용어를 맞춘다.
    assert parse_command('복구해줘') == {'type': Command.RESET}


def test_resume_keyword():
    assert parse_command('재개') == {'type': Command.RESUME}


def test_resume_keyword_synonyms():
    assert parse_command('계속해줘') == {'type': Command.RESUME}
    assert parse_command('이어서 해') == {'type': Command.RESUME}


def test_manual_move_keywords():
    assert parse_command('홈으로 가') == {'type': Command.MOVE_NAMED, 'named_target': 'home'}
    assert parse_command('정면을 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'front'}
    assert parse_command('위를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'up'}
    assert parse_command('아래를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'down'}
    assert parse_command('컨베이어를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'watch'}


def test_manual_move_keyword_with_surrounding_words():
    assert parse_command('로봇아 홈으로 가 줘') == {
        'type': Command.MOVE_NAMED, 'named_target': 'home'}


def test_manual_move_keywords_with_omitted_particles():
    # 사람마다 조사(을/를)를 생략하거나 '정면' 대신 '앞'처럼 다른 단어를 쓰기도 한다.
    assert parse_command('정면 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'front'}
    assert parse_command('앞을 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'front'}
    assert parse_command('앞 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'front'}
    assert parse_command('위 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'up'}
    assert parse_command('아래 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'down'}
    assert parse_command('밑을 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'down'}
    assert parse_command('컨베이어 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'watch'}
    assert parse_command('벨트를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'watch'}
    assert parse_command('집으로 가') == {'type': Command.MOVE_NAMED, 'named_target': 'home'}


def test_mode_switch_keywords():
    assert parse_command('수동 모드로 전환해줘') == {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}
    assert parse_command('자동 모드로 전환해줘') == {'type': Command.MODE_SWITCH, 'mode': Mode.AUTO}


def test_mode_switch_without_mode_word():
    # "모드"/"전환"/"변환" 없이 "~으로 해"만으로도 전환되도록 완화했다.
    assert parse_command('자동으로 해줘') == {'type': Command.MODE_SWITCH, 'mode': Mode.AUTO}
    assert parse_command('수동으로 바꿔') == {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}


def test_fetch_tool_keyword():
    assert parse_command('망치 갖다줘') == {'type': Command.FETCH_TOOL, 'tool': 'hammer'}
    assert parse_command('드라이버 가져다줘') == {'type': Command.FETCH_TOOL, 'tool': 'screwdriver'}


def test_fetch_tool_keyword_driver_synonyms():
    assert parse_command('십자드라이버 가져다줘') == {
        'type': Command.FETCH_TOOL, 'tool': 'screwdriver'}
    assert parse_command('일자드라이버 가져다줘') == {
        'type': Command.FETCH_TOOL, 'tool': 'screwdriver'}


def test_fetch_tool_keyword_bottle():
    assert parse_command('물병 갖다줘') == {'type': Command.FETCH_TOOL, 'tool': 'bottle'}


def test_unknown_command():
    assert parse_command('asdf') == {'type': Command.UNKNOWN, 'text': 'asdf'}


def test_tool_name_without_fetch_intent_is_unknown():
    assert parse_command('드라이버') == {'type': Command.UNKNOWN, 'text': '드라이버'}


def test_negated_fetch_command_is_unknown():
    text = '드라이버 가져오지 마'
    assert parse_command(text) == {'type': Command.UNKNOWN, 'text': text}


def test_automobile_does_not_switch_auto_mode():
    assert parse_command('자동차') == {'type': Command.UNKNOWN, 'text': '자동차'}


def test_fetch_tool_command_does_not_trigger_manual_move():
    # '컨베이어'/'위'처럼 이동 키워드와 겹치는 단어가 섞여 있어도, 이동 문구
    # 전체(조사 생략 변형 포함)가 없으면 fetch_tool로 정상 처리돼야 한다.
    result = parse_command('컨베이어 위에 있는 망치 가져와')
    assert result == {'type': Command.FETCH_TOOL, 'tool': 'hammer'}
