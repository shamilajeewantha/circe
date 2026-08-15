from setuptools import find_packages, setup

package_name = "circe_explore"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Shamila",
    maintainer_email="shamilajeewantha1@gmail.com",
    description="Coverage-aware NBV station selection for the circe rover.",
    license="MIT",
    entry_points={"console_scripts": [
        "explore_node = circe_explore.explore_node:main",
    ]},
)
