# AGC_Analysis

Adaptive Granger Causality (AGC) analysis of binary neural spiking data

This repository contains the implementation of our method, called AGC analysis, proposed in the following PNAS'18 article: 

A. Sheikhattar, S. Miran, J. Liu, J. B. Fritz, S. A. Shamma, P. O. Kanold, and B. Babadi (2018). Extracting neuronal functional network dynamics via adaptive Granger causality analysis, Proceedings of the National Academy of Sciences (PNAS), 2018 (www.pnas.org/cgi/doi/10.1073/pnas.1718154115)

and are disseminated for public use in the spirit of easing reproducibility. The experimental data sets used in this article are deposited on DRUM at http://hdl.handle.net/1903/20546 .

Implementation: the AGC analysis for the simulated set of spiking data, where the dynamic causal interactions among network of neurons are extracted from the spiking activties of neuronal ensemble. (Read the "Applications: A Simulated Example" section of article for further details.)

This code is composed of 4 main parts:  
1) Spike train generation
2) Adaptive estimation and computation stage
2.1) 2-fold even-odd cross-validation
2.2) Adaptive estimation of sparse and time-varying modulation parameter vectors associated with full and reduced GLM models from spiking responses
2.3) Recursive computation of deviance difference statistics
3) Statistical inference framework: 
3.1) Apply non-central chi-squared filtering and smoothing algorithm to deviance data
3.2) Perform the FDR control procedure based on the Benjamini-Yekutieli (BY) rejection rule
4) Plot the resulting figures
