This repository contains the code associated to the manuscript Fiorentino and Scialdone, title, bioRxiv, 2021. We study the impact of size and geometry of 2D epithelial tissues that sense a shallow gradient of an external signalling molecule through intercellular communication. To this end, we generalise the local-excitation global-inhibition (LEGI) model of multicellular gradient sensing to 2D cell configurations of different size and geometry. Moreover, beyond the standard nearest neighbour exchange (NNE) mechanism of intercellular communication, we introduce the intercellular space diffusion (ISD) model, in which the LEGI global reporter molecule is exchanged between cells diffusing in the intercellular space, thus allowing for long-range communication.

 We provide: 
- a python3 Jupyter notebook for the generation of 2-dimensional epithelial sheets with different number of cells and mean polygon number (average number of sides per cell), [ConfigGenerator.ipynb](/ConfigGenerator.ipynb); see section 1.
- a cython-based script ([ISD_Full_Simulation.py](/ISD_Full_Simulation.py)) through which parallel simulations of the ISD model can be carried out, with the associated functions; see section 2.
- a python3 Jupyter notebook for reproducing all the figures and the supplementary material contained in the manuscript, [2DLEGI_reproducibility.ipynb](/2DLEGI_reproducibility.ipynb); see section 3.

All the configurations used in the manuscript and the results of the simulations of the ISD model are provided in the [CONF](/CONF/) and [ISD](/ISD/) folders, respectively.

# Setting up a python3 virtual environment

The packages needed to set up the python3 virtual environment for reproducing the analyses are listed in the file [requirements.txt](/requirements.txt).
The virtual environment, called for instance 'epi-venv', can be created through the following commands:

```
python -m venv epi-venv
source epi-venv/bin/activate
pip3 install ipykernel
ipython kernel install --user --name=epi-venv
python -m pip install -r requirements.txt
```


# 1. Generation of epithelial cell configurations

The Jupyter notebook [ConfigGenerator.ipynb](/ConfigGenerator.ipynb) allows to generate cell configurations with a specified number of cells and mean polygon number; it is based on the python package [tyssue](https://github.com/DamCB/tyssue). 

The pipeline is the following:
- Generate a configuration with a specified number of hexagonal cells and a Gaussian noise on the position of the cell centroids; by definition this configuration has mean polygon number 6 (average number of sides per cell) and all the vertices (points in which edges meet) have order 3.

- Collapse a randomly chosen edge (cell boundary) to create a higher-order vertex.

- Repeat the edge collapse until the desired mean polygon number is reached.

The ten sets of configurations used in the manuscript are provided in the folder [CONF](/CONF/).

# 2. Numerical simulations of the ISD model

Using the 2D epithelial tissues generated in the previous section, we simulate the diffusion of the LEGI global reporter in the intercellular space and we estimate the exchange rate between each pair of cells in a given configuration.

The core function that simulates the Brownian motion of the LEGI global reporter in the intercellular space, given the cell configuration and the parameters defining the communication regime (i.e., the diffusion coefficient D of the LEGI global reporter and its internalziation rate &lambda;) is written in [cython](https://cython.readthedocs.io/en/latest/) and provided in the file [ISDBrownian.pyx](/ISDBrownian.pyx).

To turn the cython file into a compiled extension, the provided [setup](/setup.py) file is needed and the following command should be run

```
python setup.py build_ext --inplace
```

This produces the C file [ISDBrownian.c](/ISDBrownian.c) and a [shared library object](/LISBrownian.cpython-37m-x86_64-linux-gnu.so). The cython function is now ready to be imported in a python script or Jupyter notebook.

The full framework for simulating multiple trajectories in a parallel mode is contained in the python script [ISD_Full_Simulation.py](/ISD_Full_Simulation.py). To run a simulation, in this script you will need to specify:
- the folder that contains the cell configurations (e.g., [CONF](/CONF/)); 
- the number of Brownian trajectories that should be generated for each cell in each configuration;
- the list of pairs of ISD parameters [D,&lambda;] (they can be multiple to explore more than one 'communication regime');
- the number of cores that should be used (this value is bounded by the number of cores available on your machine).

Having defined the parameters, the script will save, for each cell configuration and each parameter setting, in a .ISD format the matrix of mean absorption times and the matrix of absorption probabilities. From the latter the matrix of the exchange rates between cells is computed (see section 3). The associated functions are included in the file [sim_functions.py](/sim_functions.py)

# 3. Analysis of the precision of gradient sensing and reproducibility

The Jupyter notebook [2DLEGI_reproducibility.ipynb](/2DLEGI_reproducibility.ipynb) contains the code needed to reproduce all the analyses and the figures shown in the manuscript. It uses the cell configurations from the [CONF](/CONF/) folder and the results of the simulations of the ISD model from the [ISD](/ISD/) folder.

The notebook contains the following analyses:

- Example plots of a set of 2D epithelial configurations with different number of cells and mean polygon number;
- Example of the difference between the exchange rates of the LEGI global reporter in the same cell configuration, in the two models of intercellular communication that we study in the manuscript: the nearest-neighbour exchange (NNE) and the intercellular space diffusion (ISD);
- Study of the different settings of the parameters in the NNE and ISD models, which allow us to identify different 'regimes' of intercellular communication;
- Computation of the signal-to-noise-ratio, which quantifies the precision of gradient sensing by the 2D cell configuration, generalizing the LEGI model to 2D configurations (see the associated functions in [SNRfunctions.py](/SNRfunctions.py));
- Impact of tissue size on the precision of gradient sensing, in the NNE and ISD model and different communication regimes;
- Impact of tissue geometry (i.e., changes of the mean polygon number) on the precision of gradient sensing, in the NNE and ISD model and different communication regimes;
- Comparison between the ISD and NNE communication modes for the same cell configurations.

In the same notebook we also provide all the analyses needed to reproduce the supplementary figures.
