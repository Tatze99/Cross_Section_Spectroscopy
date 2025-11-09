# Cross_Section_Spectroscopy

### How to setup the virtual environment:
- Install Python 3.12 (recommended)
- Download the repository to an arbitrary location
```
python -m venv ./venv
```
- This will install a virtual environment in the "venv" folder
- Activate the virtual environment by running activate.bat
```
./venv/Scripts/activate      # on Windows
```
- install the required packages:
```
python -m pip install -r ./requirements.txt
```
- Finally, we need to add the root folder to the system path
```
set PYTHONPATH=%CD%         # on Windows
```