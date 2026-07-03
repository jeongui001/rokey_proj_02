from setuptools import find_packages, setup

package_name = 'stt_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/stt_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='마이크 VAD + 로컬 Whisper -> /user_command/text',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt_node = stt_node.stt_node:main',
        ],
    },
)
