from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="signal2recurrence",
    version="0.1.0",
    author="Kevin [Your Last Name]",
    author_email="your.email@psu.edu",
    description="Deep metric learning for sequential signal analysis via recurrence plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/signal2recurrence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
    keywords="deep-learning metric-learning recurrence-plots signal-processing biomarkers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/signal2recurrence/issues",
        "Source": "https://github.com/yourusername/signal2recurrence",
    },
)
