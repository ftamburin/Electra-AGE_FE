# Electra-AGE_FE
Code and splits for the paper "Combining ELECTRA and Adaptive Graph Encoding for Frame Identification" at [LREC2022](https://lrec2022.lrec-conf.org/en/).

The code is in this repository, while the splits must be downloaded from this [link](http://corpora.ficlit.unibo.it/UploadDIR/GitHub/experiments.tar.gz) and extracted in the repository root folder.

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

Results obtained on a NVIDIA Titan XP 12GB and:
- CUDA 10.0
- cuDNN 7.4.2
- python 3.6.7
- pytorch 1.4.1
- transformers 3.1.0
- networkx 2.2

Parts of this package are based on [pyfn](https://github.com/akb89/pyfn).

If you find this code useful in your research, please cite:
```
@InProceedings{Tamburini:2022:LREC,
  author    = {Tamburini, Fabio},
  title     = {Combining ELECTRA and Adaptive Graph Encoding for Frame Identification},
  booktitle = {Proceedings of The 13th Language Resources and Evaluation Conference},
  month     = {June},
  year      = {2022},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
}
```

In case of problems contact me at <fabio.tamburini@unibo.it>.
