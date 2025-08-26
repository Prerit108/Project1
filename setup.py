from setuptools import find_packages,setup
from typing import List

HYOHEN_E_DOT = "-e ."  ## It is added in the requirement.txt , to link both the files.
## when we run requirement.txt file it will automatically run this file.
def get_requirements(file_path:str) -> List[str]:
    """This function will return the list of the requirements"""
    requirements =[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYOHEN_E_DOT in requirements:
            requirements.remove(HYOHEN_E_DOT)
        

    return requirements
    


setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Prerit",
    author_email="cloudsharma909@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirement.txt")


)