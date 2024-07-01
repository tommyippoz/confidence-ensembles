import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
     name='confidence-ensembles',
     version='0.6',
     scripts=[],
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unitn.it",
     description="Confidence Ensembles to Improve Classification",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/confidence-ensembles",
     keywords=['machine learning', 'confidence', 'safety', 'ensemble'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
