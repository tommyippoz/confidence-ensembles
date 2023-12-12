import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "confidence-ensembles",
    version = "0.0.1",
    author = "tommyippoz",
    author_email = "tommaso.zoppi@unifi.it",
    description = "Confidence Ensembles to improve Classification",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/tommyippoz/confidence-ensembles",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)
