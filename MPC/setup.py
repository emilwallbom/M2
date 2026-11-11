from glob import glob

from setuptools import find_packages, setup


setup(
    name="mpc_controller",
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/mpc_controller"]),
        ("share/mpc_controller", ["package.xml"]),
        ("share/mpc_controller/launch", glob("launch/*.launch.py")),
        ("share/mpc_controller/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "casadi", "matplotlib", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="emilwallbom",
    maintainer_email="emil.wallbom02@gmail.com",
    description="Hybrid A* warm start and OBCA trajectory optimization.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "mpc_node = mpc_controller.ros_node:main",
            "mpc_demo = mpc_controller.demo:main",
            "scenario_goal = mpc_controller.scenario_goal:main",
        ],
    },
)
