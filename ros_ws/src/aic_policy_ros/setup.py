from setuptools import find_namespace_packages, setup

package_name = "aic_policy_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_namespace_packages(include=["aic_policy_ros*", "vision*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/task_manifest.json"]),
    ],
    install_requires=["numpy", "opencv-python-headless", "scipy", "setuptools"],
    zip_safe=True,
    maintainer="aic-policy",
    maintainer_email="user@example.com",
    description="Placeholder insert_cable policy (ROS) for AIC",
    license="Apache-2.0",
)
