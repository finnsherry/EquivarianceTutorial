# AIM 2025: Equivariance Tutorial

## Getting started
The code for the tutorial is in this GitHub repository; to get it locally you need to have git installed ([Git for Windows](https://git-scm.com/downloads/win)).

The tutorial is implemented in Python. [Miniconda](https://www.anaconda.com/download/success) is a package manager that allows you to get Python and install all necessary packages.

Once you have both installed, navigate to the directory in which you want to save this tutorial and open it in the miniconda prompt. There, execute the command 
```
git clone https://github.com/finnsherry/EquivarianceTutorial
```
Next, navigate into the downloaded repository using
```
cd EquivarianceTutorial
```
Now, create and activate a conda environment 
```
conda create -n equivariance python=3.13
conda activate equivariance
```
You can now install all necessary packages:
```
pip install requirements.txt
```
The tutorial is a jupyter notebook, you can find it by executing
```
jupyter lab
```
Alternatively, you can open it in code editors such as Visual Studio Code.
