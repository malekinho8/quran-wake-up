from setuptools import setup, find_packages

setup(
    name="fajrGPT",
    version="1.7.2",
    author="Malek Ibrahim",
    author_email="shmeek8@gmail.com",
    description=("A Python application to assist in waking up for Fajr prayer "
                 "by providing 3 interactive verses/explanations from the Quran + ChatGPT explanations "
                 "accompanied by a soothing Islamic prayer fade-in and fade-out audio file from YouTube."),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/malekinho8/quran-wake-up",
    packages=find_packages(),
    package_data={
        # Include all MP3 files in the 'assets' subdirectory as well as txt files in fajrGPT
        'fajrGPT': ['assets/noise/*.mp3', 'assets/prompt/*.txt', 'assets/prompt/scholars/*.txt'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
    install_requires=[
        "click",
        "pygame",
        "pydub",
        "moviepy",
        "openai",
        "tqdm",
        "mutagen",
        "scipy",
        "TheQuranModule"
    ],
    entry_points={
        'console_scripts': [
            'fajrGPT=fajrGPT.wake:main',
        ],
    },
)