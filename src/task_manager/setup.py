from setuptools import find_packages, setup

package_name = 'task_manager'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/task_manager.launch.py',
            'launch/handover_system.launch.py',
        ]),
        ('share/' + package_name + '/config', ['config/task_manager_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hwangjeongui',
    maintainer_email='hwangjeongui01@gmail.com',
    description='명령 해석·상태 머신 감독 노드',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_manager_node = task_manager.task_manager_node:main',
        ],
    },
)
