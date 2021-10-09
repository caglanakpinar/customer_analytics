import setuptools
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="customeranalytics_arm64",
    version="0.0.2",
    author="Caglan Akpinar",
    author_email="cakpinar23@gmail.com",
    description="""
                Funnels, Cohorts, A/B Tests, e-commerce, CLV Prediction, Product Analytics, Churn, Anomaly Detection,
                Customer Segmentation, RFM, Statistics
                """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="""
                Funnels, Cohorts, A/B Tests, e-commerce, CLV Prediction, Product Analytics, Churn, Anomaly Detection,
                Customer Segmentation, RFM, Statistics, Customer Analytics
                """,
    packages=find_packages(exclude='__pycache__'),
    py_modules=['customer_analytics', "customeranalytics/web", "customeranalytics/docs"],
    install_requires=[
        "PyYAML",
        "psutil",
        "requests",
        "urllib3",
        "email_validator",
        "flask_login",
        "flask",
        "python-decouple",
        "flask_migrate",
        "flask_wtf",
        "sqlalchemy",
        "email-validator",
        "screeninfo",
        "pandas",
        "elasticsearch",
        "pyyaml",
        "schedule",
        "h2o",
        "psutil",
        "abtest",
        "pygeohash",
        "random2",
        "pytest-shutil",
        "python-math",
        "chart_studio"
    ],
    url="https://github.com/caglanakpinar/customer_analytics",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)