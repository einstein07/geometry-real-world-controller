from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'controller_real_world'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name), ['parameters.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sindiso',
    maintainer_email='mkhsin035@myuct.ac.za',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_real_world = controller_real_world.ringattcontroller_real_worldractor:main',
        ],
    },
)
