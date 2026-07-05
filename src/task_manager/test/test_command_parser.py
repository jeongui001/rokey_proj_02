from task_manager.command_parser import Command, Mode, parse_command


def test_stop_keyword():
    assert parse_command('멈춰') == {'type': Command.STOP}
    assert parse_command('로봇아 정지해줘') == {'type': Command.STOP}


def test_reset_keyword():
    assert parse_command('리셋') == {'type': Command.RESET}
    assert parse_command('초기화 해줘') == {'type': Command.RESET}


def test_manual_move_keywords():
    assert parse_command('홈으로 가') == {'type': Command.MOVE_NAMED, 'named_target': 'home'}
    assert parse_command('정면을 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'front'}
    assert parse_command('위를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'up'}
    assert parse_command('아래를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'down'}
    assert parse_command('컨베이어를 봐') == {'type': Command.MOVE_NAMED, 'named_target': 'watch'}


def test_manual_move_keyword_with_surrounding_words():
    assert parse_command('로봇아 홈으로 가 줘') == {
        'type': Command.MOVE_NAMED, 'named_target': 'home'}


def test_mode_switch_keywords():
    assert parse_command('수동 모드로 전환해줘') == {'type': Command.MODE_SWITCH, 'mode': Mode.MANUAL}
    assert parse_command('자동 모드로 전환해줘') == {'type': Command.MODE_SWITCH, 'mode': Mode.AUTO}


def test_fetch_tool_keyword():
    assert parse_command('스패너 갖다줘') == {'type': Command.FETCH_TOOL, 'tool': 'spanner'}
    assert parse_command('드라이버 가져다줘') == {'type': Command.FETCH_TOOL, 'tool': 'driver'}


def test_fetch_tool_keyword_water_bottle():
    assert parse_command('물병 갖다줘') == {'type': Command.FETCH_TOOL, 'tool': 'water_bottle'}


def test_unknown_command():
    assert parse_command('asdf') == {'type': Command.UNKNOWN, 'text': 'asdf'}
