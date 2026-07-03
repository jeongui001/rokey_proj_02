from setuptools import find_packages, setup

package_name = 'handover_ui'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/handover_ui.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='PyQt 데스크톱 UI (rosbridge 경유)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handover_ui = handover_ui.main:main',
        ],
    },
)
