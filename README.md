# Cross_Section_Spectroscopy

Cross Section Spectroscopy is a python based graphical user interface (GUI) which can be used to evaluate spectroscopic absorption and fluorescence measurements of a laser material. 

### Author: 
Martin Beyer: m.beyer@uni-jena.de


## How to add a new measurement
In the ```measurements``` folder you can create a new folder with a name corresponding to the date of the measurement and the name of the material. In the folder you should place the following files: 
1. ```*absorption*.txt```
2. ```*reference*.txt```
3. ```*fluorescence*.txt```

So far, only ```txt``` files are supported. The file name before and after the ```*keyword*``` can be arbitrary. In the absorption file detection function, the ```*reference*``` keyword's appearence is forbidden, therefore a name of ```*absorption_reference*``` will be correctly recognized as the reference file. 

4. ```basedata.json```

The basedata file has the following format:
```
{
    "name": "material",                                     # display name  
    "length": 1.2e-2,                                       # thickness/length in m
    "tau_f": 1.1e-3,                                        # fluorescence lifetime in s
    "N_dop": 6e26,                                          # doping concentration in 1/m³
    "n": 1.55,                                              # refractive index
    "temperature": 295,                                     # temperature in K
    "energy_lower_level": [0, 233, 372, 595],               # wave number in 1/cm
    "energy_upper_level": [10232,10511,10833],
    "ZPL": 977.3e-9                                         # zero phonon line wavelength in m
}
```
Note that the ```energy_lower_level``` and ```energy_higher_level``` keywords are optional. If they are not given, their standard value has one entry with the upper level given by the numerical value of the zero phonon line (ZPL). The comments should not be added in the .json file, as this breaks the format.


## How to setup the virtual environment:
- Install Python 3.14 (recommended)
- Download the repository to an arbitrary location
```
git clone https://github.com/Tatze99/Cross_Section_Spectroscopy  # or use the Github Desktop application
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


## Program explanation

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./ui_images/GUI_screenshot.png">
    <img src="./GUI_screenshot.png">
  </picture>
</p>

- We can display the fluorescence and absorption measurement and calculate and save the absorption and emission cross sections 
- the switches ```Use McCumber``` and ```Use Füchtbauer``` are for using the respective schemes to calculate the emission cross sections either from the absorption measurement (McCumber) using the reciprocity relation or the fluorescence measurement (Füchtbauer-Ladenburg equation)

### Config Crystal
- With the switch ```Config Crystal``` you can manipulate the values from the basedata.json file to change the material properties like doping concentration or length/thickness. 

### Config Fluorescence
- With the switch ```Config Fluorescence``` you can apply a Fourier filter to the fluorescence data.

### Config Absorption
- With the switch ```Config Absorption``` you can apply a Fourier filter to the raw data and/or a Savitzky Golay filter to the calculated absorption cross section.
- Furthermore we can precisely control the referencing to the reference measurement. We assume that we have at least two points where the absorption is zero and the absorption measurement and reference measurement should coincide. The two wavelengths can be chosen by adjusting ```zero wavelength 1/2```. We can furthermore add a bandwidth to these zero-absorption zones, then a 4th-order polynomial is used to calibrate the reference data to the absorption measurement

### Config Cross Sections
- With the switch ```Config Cross Sections``` you customize the calculation of the emission cross sections with McCumber or Füchtbauer-Ladenburg (FL). You can activate ```Average McCumber``` to obtain an average value of the emission cross section between the McCumber relation and Füchtbauer-Ladenburg method. As McCumber fails to yield reliable results at wavelength ranges with low absorption, we use Füchtbauer-Ladenburg above the ```MC central WL``` range. Vice versa, Füchtbauer-Ladenburg yields false results for wavelength ranges with a large absorption cross sections, as here reabsorption effects weaken the fluorescence signal. We can now smoothly interpolate between both methods, where the interpolation range is specified with ```average bandwidth``` given in nm. 
- Finally, we can add a reabsorption correction factor to the Füchtbauer-Ladenburg method by changing the value of ```absorption depth```. 

### Save the data
- You can either save the image or the data by specifying an image format or pdf to generate an image. If you specify a text file-format like .txt or .csv, all lines from the current image will be written into a single file.