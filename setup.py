from setuptools import setup, find_packages

setup(
    name="eyewear_counter",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",        
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "aiohttp",
        "tqdm",
        "xlsxwriter",
        "nest_asyncio",
        "requests",
        "gradio",
    ],
    entry_points={
        "console_scripts": [
            "eyewear-counter-app=app.app:main",
        ],
    },
    author="Ekaterina Solovyeva",
    author_email="qksolov@gmail.com",
    description="Fast model for counting eyewear types on faces in large sets of images.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qksolov/eyewear-counter",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
