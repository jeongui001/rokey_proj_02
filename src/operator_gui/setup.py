from setuptools import find_packages, setup

package_name = 'operator_gui'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/operator_gui.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='PyQt 오퍼레이터 GUI (rosbridge 경유, handover_ui에서 분리된 확장 버전)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'operator_gui = operator_gui.main:main',
        ],
    },
)
