import os

# 이 환경에는 PyQt5와 PySide6가 모두 설치되어 있다. pytest-qt가 두 바인딩을 함께
# 로드하면(autodetect) 같은 프로세스에서 충돌해 즉시 크래시하므로, 이 프로젝트가
# 사용하는 PyQt5로 명시적으로 고정한다.
os.environ.setdefault('PYTEST_QT_API', 'pyqt5')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
