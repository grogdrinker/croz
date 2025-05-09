#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(

     name='croz',
     zip_safe=False,
     include_package_data=False,

     version='0.0.1',

     author="Gabriele Orlando",

     author_email="gabriele.orlando@umontpellier.be",

     description="A method based on pyuul to score a protein (or other biological structure) model into a cryoEM electrondensity",

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/grogdrinker/croz",
     
     packages=['croz',"croz.src"],
     package_dir={'croz': 'croz/',"croz.src":"croz/src"},

     install_requires=["torch","numpy","biopython", "mrcfile","pyuul","matplotlib",'madrax @ git+https://bitbucket.org/grogdrinker/madrax/'],

     classifiers=[

         "Programming Language :: Python :: 3",

         "Operating System :: OS Independent",

     ],

 )
