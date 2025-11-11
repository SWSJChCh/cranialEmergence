# The spontaneous emergence of leaders and followers in an agent-based model of cranial neural crest cell migration

## Overview
This repository contains Python scripts used to generate data for plots in the publication:
_The spontaneous emergence of leaders and followers in an agent-based model of cranial neural crest cell migration_

### Code Authors
- Samuel Johnson

### Date
- 11/11/2025

### Requirements
- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.57.0
- imageio==2.37.0
- imageio-ffmpeg==0.6.0
- kiwisolver==1.4.8
- llvmlite==0.44.0
- matplotlib==3.10.1
- numba==0.61.2
- numpy==2.2.4
- opencv-python==4.11.0.86
- packaging==24.2
- pillow==11.2.1
- psutil==7.0.0
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- scipy==1.15.2
- six==1.17.0

The required libraries can be installed from the requirements file using pip:

```bash
pip install requirements.txt
```

### Script Descriptions

#### VEGF.py
`VEGF.py` contains a forward-Euler solver used to update the chemoattractant profiles within the growing 
simulation domain. 

#### growthFunction.py 
`growthFunction.py` includes functions that fit _in vivo_ data of the domain length to a logistic curve, and returns
a time-resolved list of domain lengths for use in the main simulation. 

#### insertCell.py 
`insertCell.py` includes functions that creates cell objects. 

#### moveCell.py 
`moveCell.py` includes functions for cell movement according to chemical cues and a pairwise Hookean interaction potential.  

#### runSimulation.py 
`runSimulation.py` runs the main simulation and outputs a video or a .txt containing simulation data. 

### Execution 
Code is executed using the runSimulation.py file along with parameters given as command line arguments: 

```bash
python runSimulation.py spring-strength active-strength degradation-rate
```

- **spring-strength** - Scaling parameter for inter-cellular attraction (float)
- **active-strength** - Scaling parameter for active polarity-induced force (float)
- **degradation-rate** - Degradation rate of VEGF (/h)
