from setuptools import find_packages, setup
from typing import List
# find_packages: finds all the packages in the entire ML application in the directory
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list of requirements
    '''
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='MLproject',
    version='0.0.1', # update this version when next version comes
    author='Amall',
    author_email='iammuhammedamal@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt') #['pandas','numpy','seaborn'], # automatically install all libraries mentioned
)