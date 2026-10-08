from setuptools import find_packages, setup

package_name = 'thoth_vision'

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
    maintainer='habiba',
    maintainer_email='habiba@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
        entry_points={
        'console_scripts': [
            'visitor_perception_node = thoth_vision.visitor_perception_node:main',
        ],
    },
)
