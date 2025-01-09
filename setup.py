from distutils.core import setup
import setuptools

setuptools.setup(
    name="confify",
    packages=setuptools.find_packages(),
    version="0.0.1",
    license="MIT",
    description="Confify is a fully typed, plug-and-play configuration library for Python.",
    author="Milin Kodnongbua",
    author_email="mil.millin@gmail.com",
    url="https://github.com/milmillin/confify",
    keywords=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)