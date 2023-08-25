from setuptools import setup, find_packages
from typing import List

# get requirements from requirements.txt
def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.strip() for r in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
        
    return requirements

setup(
    name='ZayedML',
    version='0.1',
    author='Zayed',
    author_email='azayed325@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)