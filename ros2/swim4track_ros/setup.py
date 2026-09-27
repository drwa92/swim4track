from glob import glob
from setuptools import find_packages, setup

setup(
    name="swim4track_ros", version="0.2.0", packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/swim4track_ros"]),
        ("share/swim4track_ros", ["package.xml"]),
        ("share/swim4track_ros/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="Waseem Akram", maintainer_email="waseem.akram@fer.hr",
    description="Direct-thruster policy inference for Stonefish BlueROV2 Heavy",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "controller = swim4track_ros.controller_node:main",
        "demo_watch = swim4track_ros.demo_watch:main",
    ]},
)
