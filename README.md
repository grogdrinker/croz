![logo](docs/_static/logo_small.png)

---

# About croz

## What is Croz?
Croz is a tool to optimize and score the superimposition of a cryoEM map and a PDB atomic model. It does not require any threshold, and it works directly from the cryoEM map and the PDB file.

## What do I need to use Croz?
The only requirement is that the model is roughly placed in the corresponding part of the map it refers to. "Roughly" is enough, since the optimization will take care of the rest.

## How does Croz work?
Croz uses pyuul to compare the expected and experimental occupancy of the protein, and performs a gradient descent-based optimization using PyTorch on the atomic model to maximize the match between theoretical and experimental occupancies.

Croz can move the model as a whole, but it also includes an option to rotate side chain positions.

## What else should Rob know?
There is a hidden option to move the backbone as well, but it is currently disabled in the user interface because I never managed to get good results with it. Let me know if you want me to activate it.

## citing (in case we write a paper about this. To not forget)
If you use Croz in your research, please consider citing:

## Installation
You can install Croz with pip building it locally.

To install Croz locally from source:

1. Navigate to the directory of croz. For example:
   ```bash
    cd croz
2. Install the package using pip:
   ```bash
    pip install .
3. you can test the functions of Croz using the script "croz/run_testing.py". It is basically a tutorial

# Usage

## Using Croz into a python script

Croz can be imported as a python module

```python
    from croz.croz import run_optimization
    test_pdb = "your favouritePDBmodel.pdb"
    test_map = "the_respectiveCryoEM_mapFile.mrc"

    score = run_optimization(test_pdb,test_map,verbose=True)
```
the score  is a Pearson's correlation coefficient. The higher the better

## Help

For bug reports, features addition and technical questions please contact gabriele.orlando@umontpellier.fr
