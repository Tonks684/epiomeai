# ML Engineer Take-Home Task

## Task 
Develop a machine learning model to predict Alzheimer's disease progression from DNA methylation data as two independent binary classification tasks. 
You will be provided with an HDF5 file containing a pre-processed feature matrix as your starting point. 

- **Task 1:** Control → Control vs. Control → Mild Cognitive Impairment (MCI)
- **Task 2:** MCI → MCI vs. MCI → Alzheimer's Disease

For each task, the model input at each time step is a DNA methylation profile represented as a two-dimensional matrix (samples × features). 
The prediction target is a binary variable indicating conversion (1) or non-conversion (0).

You are encouraged to experiment with different architectures — including but not limited to convolutional or recurrent neural networks — and to justify your choices. 
If you use a complicated deep learning architecture make sure you implement a simple regression model to compare performance. 

## Dataset

The feature matrix is derived from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/) study and 
contains 1,905 DNA methylation samples collected from peripheral blood across 649 unique individuals. 
The 2,000 features are pre-selected based on the variance ratio of beta values between converters and non-converters.

## Deliverable

A GitHub repository containing all code, with a README covering:

- Environment setup and dependencies
- How to run model training
- How to run validation, including all modelling decisions and hyperparameter choices
- How to run model evaluation and interpret outputs

## What We're Looking For

- Solid implementation of the architecture and training procedure
- Reproducibility: someone else should be able to clone the repo and run it end to end
- Clear, well-structured code with appropriate documentation

## Notes

- You may use standard deep learning libraries (PyTorch or TensorFlow)
- You may reference existing implementations for standard components, but the model architecture and model evaluation should be your own
- Include a brief written reflection on what you would improve or do differently with more time

---

**Feel free to use AI tools or not for this task. 
If you do, please explain what you used them for, how you used them, and what you were able to achieve with them.**