 The first step in running the code is ensuring the necessary libraries are installed. In this the libraries to be installed is numpy and matplotlib. Numpy aids in carrying out computations in python. Matplotlib is a python package that is used to visualize that is it makes it possible for the wave to be displayed.  The packages are installed pip install package name pip install numpy.  
       After installing the modules the following steps are used to run the codes. 
    1. Ensure the code files are stored in the same folder.
    2.  Open the files in a python IDE.
    3. Since the task 5.py and task 6.py are implemented using the test.py code file, the file to be run will be test.py. 
    4. Run the test.py file. The output is waveform.
Since the project is mathematically based some functions are used to help in computation of certain values. Such functions include Fourier transform, cutoff functions and, convergence rates.
Fourier transformation
Fourier transformation is a mathematical function that breaks down a function into its component frequencies. The function is usually a time function or a signal function example sound waves.  Fourier transform is a mathematical approach that converts a time function into a frequency function.  Fourier transform breaks down the shape of a wave into sinusoids. 	
The function computes the frequency of complex numbers, these numbers are non-negative numbers in which there sign is not put into consideration. The non-negative numbers represent the value of a specific frequency present in the initial function and whose argument is the 
difference between two phase of periodic waves being not equal to zero.   
The functions in the code files serve different purposes. This functions are mathematical hence the need to use numpy for computation arises.
Running the first parts
1. Install conda 1
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
2. For convenience, create link to the conda binary
ln -s ~/miniconda3/bin/conda .local/bin
3. Prevent conda from activating the base environment by default 2
conda config --set auto_activate_base false
4. Create environment called fdm (as in finite difference methods) for the project
conda create -n fdm python=3.7
5. Install NumPy
conda install -n fdm numpy
6. Install matplotlib for plotting
conda install -n fdm matplotlib
7. For running unit tests, install pytest
conda install -n fdm pytest
8. For convenience, install IPython shell
conda install -n fdm ipython
9. Activate the fdm environment 3
conda activate fdm
