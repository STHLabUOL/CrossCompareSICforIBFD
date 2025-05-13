# Cross-Comparison of Neural Architectures and Data Sets for Digital Self-Interference Modeling

This respository contains the scripts and notebooks used in [1] for training and evaluating models for digital self-interference cancellation. Note that this includes scripts and code snippets from [\[3\]](https://github.com/abalatsoukas/fdnn).

For guidance on how to execute Jupyter-Notebooks, please refer to the official [Documentation](https://docs.jupyter.org/en/latest/).


## 1. Datasets

All three datasets reside in the <i>data/</i> directory. As explained in [1], the real data was sourced from the author's repository [\[3\]](https://github.com/abalatsoukas/fdnn). The synthetic Hammerstein and Wiener data was generated according to [\[2\]](https://github.com/STHLabUOL/SICforIBFD), where the following default parameters were modified:

| Parameter | Value |
| -------- | ------- |
| cfgHT.MCS | 2 |
| cfgHT.PSDULength | 4*1000 |
| parSigs.SINR_dB_awgn | 62 |

For convenience, this repository includes the Matlab scripts from [\[2\]](https://github.com/STHLabUOL/SICforIBFD) with the appropriate modifications already applied in the directory <i>synth_data_gen/</i>.



## 2. Model Training

Training of the Hammerstein, Wiener and WienerHammerstein model is handled by <b>1_training_models_H_W_WH.ipynb</b>, while training of the FFNN model is handled separately in <b>1_training_model_FFNN.ipynb</b>. Both notebooks are structured similarly.

In the notebooks, you may specify the type of data used, what model architecture to train, and whether or not to include linear preprocessing of the data. Additionally, export of results for later retrieval in <b>summarize_SIC_results.ipynb</b> can be enabled. Upon execution of the notebook, you will be prompted to confirm the data export.

## 3. SIC Performance Summary

Previously saved results are summarized in <b>2_summarize_SIC_results.ipynb</b>, together with the comparison of the data's power spectral densities. The default path points to that data of which the results were published in [1].

## 4. Number of Parameters and GMACs Comparison
Finally, a brief comparison of number of parameters and Giga-Multiply-Accumulate operations (GMACs) between models is carried out in <b>3_compute_numParams_and_GMACs.ipynb</b>.

## References

[1] G. Enzner, N. Knaepper, A. Chinaev, "Cross-Comparison of Neural Architectures and Data Sets for Digital Self-Interference Modeling", 2025<br>
[2] G. Enzner, A. Chinaev, S. Voit, A. Sezgin, "On Neural-Network Representation of Wireless Self-Interference for Inband Full-Duplex Communications", Submitted to IEEE Int. Conf. on Acoust., Speech and Signal Process., 2025<br>
[3] A. Balatsoukas-Stimming, "Non-linear digital self-interference cancellation for in-band full-duplex radios using neural networks," in IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), Jun. 2018

## Acknowledgment
This work is funded by [German Research Foundation (DFG)](https://asn.uni-paderborn.de/) - Project 449601577

## Citation
If you use this code or our dataset, please cite our paper:
```
@inproceedings{Enzner2025ibfdCrossCompare,
  title={Cross-Comparison of Neural Architectures and Data Sets for Digital Self-Interference Modeling},
  author={Enzner, Gerald and and Knaepper, Niklas and Chinaev, Aleksej},
  booktitle={---},
  month={May.},
  year={2025}
}
```

