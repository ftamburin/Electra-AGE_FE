# Electra-AGE_FE
Code and splits for the submitted paper.

The code is in this repository, while the splits must be downloaded from this [link](http://corpora.ficlit.unibo.it/UploadDIR/experiments.tar.gz) and extracted in the repository root folder.

For reproducing the experiments:
1) Create ELECTRA embeddings for a given split (here FN1.5\_08):
```
    cd scripts
    bash ./CreateVectors.sh 1.5_08
```
2) Perform the entire set of 10 experiments with different random initialisation:
```
    bash ./FrameID.sh -m train -x 1.5_08 -v 1.5 -e electra 
```
3) Look at the final detailed results on the test set:
```
    python3 ./showResults.py ../experiments/xp_1.5_08/frameid/results/summary
```
