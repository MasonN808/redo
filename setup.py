from setuptools import setup, find_packages

# setup(
#     name='redo',  # Replace with your project's name
#     version='0.1.0',  # Initial version
#     packages=find_packages(),  # Find packages in the 'src' directory
# )
setup(
    name='redo',  # Replace with your project's name
    version='0.1.0',  # Initial version
    packages=find_packages(where='src'),  # Find packages in the 'src' directory
    package_dir={'': 'src'},  # Tell setuptools that packages are under 'src'
)