from glob import glob

from setuptools import setup

package_name = "eeg_workload_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/data", glob("data/*.csv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Afroza Islam",
    maintainer_email="afrozaislam.mte18@gmail.com",
    description="Simulated robot that adapts its speed to an EEG workload decoder.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "decoder_replay = eeg_workload_robot.decoder_replay:main",
            "speed_controller = eeg_workload_robot.speed_controller:main",
        ],
    },
)
