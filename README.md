# Cross-Comparison of Neural Architectures and Data Sets for Digital Self-Interference Modeling

This respository contains the scripts used in [1] for training and evaluating models for digital self-interference cancellation. Note that this includes code snippets, as well as complete scripts sourced from [\[2\]](https://github.com/abalatsoukas/fdnn).



## Global Configuration

Modify <b>config.py</b> to specify paths pointing to the three datasets, real, synthetic-hammerstein, and synthetic-wiener.


## Model Training

Training of the Hammerstein, Wiener and WienerHammerstein model is handled by <b>training_models_H_W_WH.ipynb</b>, while training of the FFNN model is handled separately in <b>training_model_FFNN.ipynb</b>. Both notebooks are structured similarly.

In the notebooks, you may specify the type of data used, what model architecture to train, and whether or not to include linear preprocessing of the data. Additionally, export of results for later retrieval in <b>summarize_SIC_results.ipynb</b> can be enabled. Upon execution of the notebook, you will be prompted to confirm the data export.

## Review and Comparison

Previously saved results are summarized in <b>summarize_SIC_results.ipynb</b>, together with the comparison of the data's power spectral densities. This repository includes the data published in [1]. You may configure the path to point to your own export files.


## References

[1] G. Enzner, N. Knaepper, A. Chinaev, "Cross-Comparison of Neural Architectures and Data Sets for Digital Self-Interference Modeling", 2025<br>
[2] A. Balatsoukas-Stimming, "Non-linear digital self-interference cancellation for in-band full-duplex radios using neural networks," in IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), Jun. 2018

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

