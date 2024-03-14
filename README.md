## Run from source
To run pygeoutil by setting up the running environment with the source, follow these steps:

1. Clone the repository.
```
git clone https://github.com/eugenegesdisc/pygeoutil.git
```
2. Change to the cloned source directory and create the conda environment.
```
cd pygeoutil
conda env create -f conf/environment.yml
conda activate py311geoutil
```
3. Run with pytest.
```
python -m pygeoutil --help
```