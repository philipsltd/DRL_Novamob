from setuptools import find_packages, setup

package_name = 'novamob_gym'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'gymnasium',
        'numpy',
        'rclpy',
    ],
    zip_safe=True,
    maintainer='filipe',
    maintainer_email='feduardomorais@gmail.com',
    description='A custom OpenAI Gym environment for interacting with a robot in Gazebo using ROS 2.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'novamob_env = novamob_gym.novamob_env:main'
        ],
    },
)