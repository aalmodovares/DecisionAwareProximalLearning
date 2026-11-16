# Third-party code and data

The `DecisionAwarePL/vendors/` directory contains selected source files from the projects below. The implementations in `DecisionAwarePL/src/` include adaptations for the experiments in this repository. The vendor directories are subsets of the original projects; consult the upstream repositories for the complete implementations and their documentation.

| Directory | Source | License information |
| --- | --- | --- |
| `DeCaFlow_modified` | [DeCaFlow](https://github.com/aalmodovares/DeCaFlow), Almodóvar et al. | GPL-3.0; original license included in the vendor directory. The local version is modified. |
| `DeepFeatureProxyVariable` | [DeepFeatureProxyVariable](https://github.com/liyuan9988/DeepFeatureProxyVariable), Xu et al. | MIT; original license included in the vendor directory. |
| `DR-Kernel-Proxy` | [Doubly-Robust-Kernel-Proxy-Variable-Algorithm](https://github.com/BariscanBozkurt/Doubly-Robust-Kernel-Proxy-Variable-Algorithm), Bozkurt et al. | No license file was identified in the source repository. |
| `FTE-proximal` | [FTE-proximal](https://github.com/zlong66/FTE-proximal), Long and Zhang | No license file was identified in the source repository. |
| `kernel_proxies` | [kernel_proxies](https://github.com/yuchen-zhu/kernel_proxies), Mastouri et al. | MIT; original license included in the vendor directory. |
| `Neural-Moment-Matching-Regression` | [Neural-Moment-Matching-Regression](https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression), Kompa et al. | Original license included in the vendor directory. |
| `OOSR` | Zou et al., [Counterfactual Prediction for Outcome-Oriented Treatments](https://proceedings.mlr.press/v162/zou22a.html) | The source code does not have a license published on GitHub. |
| `PCL_Continuous_Treatment` | [PCL_Continuous_Treatment](https://github.com/yezichu/PCL_Continuous_Treatment), Wu et al. | MIT; original license included in the vendor directory. |

Each included license applies to its corresponding third-party code. For repositories without an identified license, consult the original source and its authors about the applicable terms.

The semi-synthetic benchmark uses TCGA data made available through the [SCIGAN repository](https://github.com/ioanabica/SCIGAN), associated with Bica, Jordon and van der Schaar, *Estimating the Effects of Continuous-valued Interventions using Generative Adversarial Networks* (NeurIPS 2020). The dataset is not included in this repository.
