# ismodRSW_satDA
## An idealized 1.5-layer isentropic model with convection and precipitation for satellite data assimilation research

This repository contains the relevant source code and documentation for the research papers: 

```Cantarello et al (2022): An idealised 1.5-layer model with convection and precipitation for satellite data assimilation research. Part I: model dynamics;```
```Bokhove et al (2022): An idealised 1.5-layer model with convection and precipitation for satellite data assimilation research. Part II: model derivation;```

both accepted for publication on the Journal of the Atmosheric Sciences (Early Online Release: <a href='https://journals.ametsoc.org/view/journals/atsc/aop/JAS-D-21-0022.1/JAS-D-21-0022.1.xml'>Part I</a>; <a href='https://journals.ametsoc.org/view/journals/atsc/aop/JAS-D-21-0023.1/JAS-D-21-0023.1.xml'>Part II</a>). It also includes the code used in Luca Cantarello's PhD (Cantarello 2021, available <a href='https://etheses.whiterose.ac.uk/29672/'>here</a>) which uses a revised version of the idealised fluid model developed by Tom Kent (modRSW; Kent et al. 2017) and a Deterministic Ensemble Kalman Filter (DEnKF) to run satellite data assimilation experiments. This README contains sufficient instruction for users to download, implement and adapt the source code, which briefly comprises Python3 scripts for the numerical solver for the discretised model, data assimilation algorithms, plotting and data analysis. 

For any questions or code bugs please send an email to mmlca@leeds.ac.uk.

## References
#### Thesis + articles:

* Cantarello, L. (2021): Modified shallow water models and idealised satellite data assimilation. *PhD thesis, University of Leeds*. Available at [http://etheses.whiterose.ac.uk/29672/](http://etheses.whiterose.ac.uk/29672/).

* Cantarello, L., Bokhove, O., Tobias, S.M. (2021): An idealised 1.5-layer model with convection and precipitation for satellite data assimilation research. Part I: model dynamics. *Journal of the Atmospheric Sciences*, accepted for publication. [DOI](https://journals.ametsoc.org/view/journals/atsc/aop/JAS-D-21-0022.1/JAS-D-21-0022.1.xml).

* Bokhove, O., Cantarello, L., Tobias, S.M. (2021): An idealised 1.5-layer model with convection and precipitation for satellite data assimilation research. Part II: model derivation. *Journal of the Atmospheric Sciences*, accepted for publication. [DOI](https://journals.ametsoc.org/view/journals/atsc/aop/JAS-D-21-0023.1/JAS-D-21-0023.1.xml).

* Kent T., Cantarello, L., Inverarity, G.W., Tobias, S.M., Bokhove, O. (2020): Idealised forecast-assimilation experiments for convective-scale Numerical Weather Prediction. *EarthArXiv*, [DOI](https://eartharxiv.org/repository/view/1921/). 

## Getting started
### Versions -- Check!!
All of the source code is written in Python and relies heavily on the numpy module, amongst others. The plotting routines require matplotlib. The versions used in this development are tabled below. Other versions may work of course, but have not been tested by the authors.

Software      | Version
------------- | -------------
Python  | 3.8.3
Matplotlib  | 3.2.2
Numpy  | 1.18.5

To check python version, from the terminal:
```
python --version
```

To check numpy version, open python in the terminal, import it and use the version attribute:
```
>>> import numpy
>>> numpy.__version__
```
Same for all other modules. 

### Download and install

Clone from terminal (recommended):
* Go to the directory where you want to save the repository and use the command:
```
git clone https://github.com/modRSW-convective-scale-DA/ismodRSW_satDA.git
```
* Once downloaded, to get any updates/upgrades to the original clone, use the command:
```
git pull https://github.com/modRSW-convective-scale-DA/ismodRSW_satDA.git
```

Direct download: 
* click on the download link on the repository url [https://github.com/modRSW-convective-scale-DA/ismodRSW_satDA](https://github.com/modRSW-convective-scale-DA/ismodRSW_satDA) and save to desired directory.

### Running the code: basics

All python scripts of the repository can be executed from terminal:
```
python name_of_script.py config/config_file.py [additional arguments]
```
To kill at any point, press ```Ctrl+c```, or kill the active processes using ```top``` from the terminal.

## Brief overview of files
Here is an overview of the files contained in this repository and what they do. They are listed in the order they need to be modified or run.

### Configuration file and look-up table
* *config/config_example.py*: this file contains all the parameters, file paths and values used for running the ismodRSW model, creating the observing system and  setting up the data assimilation algorithm.
* *inversion_tables/generate_sig_eta.py*: this script should be run as soon as the configuration file has been set up.

### Model only
* *run_ismodRSW.py*: ...
* *hovmoller.py*: ...

### Assimilation framework
* create_truth+obs.py
* offlineQ.py
* main_p.py

### Plotting and data analysis
* plot_func_t.py
* plot_func_x.py
* plot_forec_x.py
* compare_stats.py
* run_ismodRSW_EFS.py
* EFS_stats.py
* err_doub_hist.py
