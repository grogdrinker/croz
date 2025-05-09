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
Package installation should only take a few minutes with any of these methods (pip, source).

### Installing Xenusia with pip:

We suggest to create a local conda environment where to install xenusia. it can be done with:

```sh
conda create -n xenusia
```
and activated with

```sh
conda activate xenusia
```

or

```sh
source activate xenusia
```

We also suggest to install pytorch separately following the instructions from https://pytorch.org/get-started/locally/

```sh
pip install xenusia
```

The procedure will install xenusia in your computer.

### Installing Xenusia from source:

If you want to install Xenusia from this repository, you need to install the dependencies first.
First, install [PyTorch](https://pytorch.org/get-started/locally/) separately following the instructions from https://pytorch.org/get-started/locally/.

Then install the other required libraries:

```sh
pip install numpy scikit-learn requests
```

Finally, you can clone the repository with the following command:

```sh
git clone https://github.com/grogdrinker/xenusia/
```

## Usage

the pip installation will install a script called xenusia_standalone that is directly usable from command line (at least on linux and mac. Most probably on windows as well if you use a conda environemnt).

### Using the standalone
The script can take a fasta file or a sequence as input and provide a prediction as output

```sh
xenusia_standalone AWESAMEPRTEINSEQENCEASINPT
```

or, for multiple sequences, do

```sh
xenusia_standalone fastaFile.fasta
```

To write the output in a file, do

```sh
xenusia_standalone fastaFile.fasta -o outputFilePath
```

### Using Xenusia into a python script

Xenusia can be imported as a python module

```python
from xenusia.run_prediction import predict
proteinSeq1 = "ASDASDASDASDASDASDDDDASD"
proteinSeq2 = "ASDADDDDDDDDDDDDDASDASDDDDASD"
proteinSeq2 = "ASDADFFFFFFFFFDDDDDDDDFFFFFFFFFASD"
inputSequences = {"ID1":proteinSeq1,"ID2":proteinSeq2,"ID3":proteinSeq3}

xenusia_output = predict(inputSequences) # which is a dict containig the predictions

```


## Help

For bug reports, features addition and technical questions please contact gabriele.orlando@kuleuven.be
