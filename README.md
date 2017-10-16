# AGC_Analysis

Adaptive Granger Causality (AGC) analysis of binary neural spiking data

This repository contains the MATLAB codes for the AGC analysis of the simulated set of spiking data, where the time-varying causal interactions among neurons are detected from the spiking activties of neuronal ensemble. 

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
