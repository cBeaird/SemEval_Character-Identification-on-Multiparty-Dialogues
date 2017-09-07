"""
    Setup file with application requirements and licensing information
    Casey Beaird
"""
from distutils.core import setup
import sys

__author__ = 'Casey Beaird'

if sys.version_info[0] != 2:
    v = sys.version
    print('python version: {0}is not supported'.format(sys.version.split('\n')[0]))
    exit(0)

setup(
    name='SemEval_Character-Identification-on-Multiparty-Dialogues',
    version='0.1',
    packages=[''],
    url='https://github.com/cBeaird/SemEval_Character-Identification-on-Multiparty-Dialogues',
    license='MIT',
    author='Casey, Chase, Brandon',
    author_email='',
    description='SemEval project Task 4: Conference Resolution',
    install_requires=['conllu', 'nltk', 'tensorflow']
)
