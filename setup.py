from setuptools import setup


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name='icepinn',
    version='0.0.1',
    author='Asher Merrill',
    author_email='asher.merrill@utah.edu',
    packages=['icepinn',],
    package_dir={'': '.'},
    download_url=r'https://github.com/asher-m/icepinn',
    description=r'',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[],
    license='MIT',
    keywords='',
    url=r'',
    install_requires=[
        'basemap>=2.0',
        'basemap-data-hires>=2.0',
        'ipython>=9',
        'jupyter>=1.0.0',
        'jupyterlab>=4.0.0',
        'matplotlib>=3',
        'netcdf4>=1.7',
        'numpy>=2',
        'scipy>=1.17',
        'torch>=2.11',
        'torchaudio>=2.11',
        'torchvision>=0.26',
        'tqdm>=4.67'
    ]
)
