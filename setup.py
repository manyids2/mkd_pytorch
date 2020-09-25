import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mkd_pytorch",
    version="1.1.2",
    author="Arun Mukundan",
    author_email="arun.mukundan@gmail.com",
    description="Multiple kernel local descriptors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manyids2/mkd_pytorch",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
