from setuptools import find_packages, setup

package_name = 'robot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_control.launch.py']),
        ('share/' + package_name + '/config', [
            'config/robot_control_params.yaml',
            'config/robot_control_calibration_params.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='RobotTask 액션 서버 - 모션/서보 파지/그리퍼 실행자',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_control_node = robot_control.robot_control_node:main',
            'probe_speedl_stream = robot_control.tools.probe_speedl_stream:main',
        ],
    },
)
