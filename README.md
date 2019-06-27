# SNR

Project of classification experiments with deep neural networks for SNR

## How to run

1.   Install Miniconda for Python 3.X on your PC. Installation files can be found on [this website](https://docs.conda.io/en/latest/miniconda.html).
2.   Open a console and navigate to a directory where project files are located. 
     ``` bash
     cd /path/to/project/dir/
     ```
3.   Create a conda environment with all dependencies included in the `environment.yml` file. This will create a new environment called `snr`.
     ```bash
     conda env create -f environment.yml
     ```
4.   Wait until all needed packages are downloaded.
5.   Meanwhile, go to [kaggle documentation](https://github.com/Kaggle/kaggle-api#api-credentials) 
and perform steps described in `API credentials` section to import `kaggle.json` with your credentials.
6.   Run `jupyter notebook` in `conda` environment with this command:
     ```bash
     conda activate snr
     jupyter notebook
     ```
7.   A new window should open in your browser. From the directory tree select project file, you want to run.


**Steps 1-5 are required only on a first run.**