# Cross_Section_Spectroscopy

Cross Section Spectroscopy is a python based graphical user interface (GUI) which can be used to evaluate spectroscopic absorption and fluorescence measurements of a laser material. 

### Author: 
Martin Beyer: m.beyer@uni-jena.de


## How to add a new measurement
In the ```measurements``` folder you can create a new folder with a name corresponding to the date of the measurement and the name of the material. In the folder you should place the following files: 
1. ```*absorption*.txt```
2. ```*reference*.txt```
3. ```*fluorescence*.txt```

So far, only ```txt``` files are supported. The file name before and after the ```*keyword*``` can be arbitrary. In the reference file, the ```*absorption*``` keyword is filtered out, therefore a name of ```*absorption_reference*``` will be correctly recognized as the reference file. 

4. ```basedata.json```

The basedata file has the following format:
```
{
    "name": "material",
    "length": 1.2e-2,
    "tau_f": 1.1e-3,
    "N_dop": 6e26,
    "n": 1.55,
    "temperature": 295,
    "energy_lower_level": [0, 233, 372, 595],
    "energy_upper_level": [10232,10511,10833],
    "ZPL": 977.3e-9
}
```
Note that the ```energy_lower_level``` and ```energy_higher_level``` keywords are optional. If they are not given, their standard value has one entry with the upper level given by the numerical value of the zero phonon line (ZPL).


### How to setup the virtual environment:
- Install Python 3.14 (recommended)
- Download the repository to an arbitrary location
```
git clone https://github.com/tatze99/LaserSimulation.git    # or use the Github Desktop application
```
- open this folder in a python capable IDE, e.g. VisualStudioCode
- Optional: open a Terminal in this folder, create a virtual environment:
```
python -m venv ./venv
```
- This will install a virtual environment in the "venv" folder
- Activate the virtual environment by running activate.bat
```
./venv/Scripts/activate      # on Windows
source ./venv/bin/activate   # on Linux
```
- install the required packages:
```
python -m pip install -r ./requirements.txt
```
- Finally, we need to add the root folder to the system path
```
set PYTHONPATH=%CD%         # on Windows
export PYTHONPATH=$(pwd)    # on Linux
```
- run the Cross_Section_Spectroscopy.py file