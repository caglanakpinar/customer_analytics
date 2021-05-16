import setuptools
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="customer_analytics",
    version="0.0.1",
    author="Caglan Akpinar",
    author_email="cakpinar23@gmail.com",
    description="clv prediction applying with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='CLV, Customer Lifetime Value, Lifetime Prediction',
    packages= find_packages(exclude='__pycache__'),
    py_modules=['clv', 'clv/docs'],
    # TODO: will be updated
    install_requires=[
    ],
    url="https://github.com/caglanakpinar/clv_prediction",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)