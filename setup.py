from setuptools import find_packages, setup
setup(
    name='infect_predict',
    packages=find_packages(),
    version='0.1.0',
    description='Infection predictor',
    author='Case predictor',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)