import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gquant", # Replace with your own username
    version="0.0.1",
    author='GuQiangJS',
    author_email='guqiangjs@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GuQiangJS/gquant",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)