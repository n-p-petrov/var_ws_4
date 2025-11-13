from setuptools import find_packages, setup

package_name = "apriltag_detector"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nppetrov",
    maintainer_email="nikola.petrov@student.uva.nl",
    description="TODO: Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "apriltag_detector = apriltag_detector.apriltag_detector:main",
            "apriltag_visualizer = apriltag_detector.apriltag_visualizer:main",
        ],
    },
)
