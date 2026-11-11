from setuptools import find_packages, setup

package_name = 'low_level_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='davax',
    maintainer_email='david.axelsson@liu.se',
    description='Low-level car controller for mapping normalized control input [0, 1] to velocity.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'low_level_controller = low_level_controller.low_level_controller:main'
        ],
    },
)
