# Classical Machine Learning Models

This repository contains simulations of ARIMA and the following 7 classical machine-learning models applied to regression problems:

- k-nearest neighbors;
- decision tree;
- random forest;
- support vector machine;
- least squares support vector machine;
- gradient boosting;
- LightGBM.

The ARIMA parameters are defined based on Akaike's Information Criterion (AIC), and the hyper-parameters of the machine-learning models are optimized using grid search with cross-validation.

You can find more information about those models at the following link: https://kaikealves.weebly.com/classical-models.html.

This repository applies the cited models to benchmark series (Mackey-Glass, Nonlinear, and Lorenz Attractor), financial series (NASDAQ, S&P 500, and TAIEX), and finally, solar energy datasets.

The files are explained as follows:

Folder [Functions](https://github.com/kaikerochaalves/ClassicalModels/tree/cd0dd870b1af03dc915be2a81f83728f179194a7/Functions) contains the libraries to generate the benchmark time series.

Folder [Graphics](https://github.com/kaikerochaalves/ClassicalModels/tree/cd0dd870b1af03dc915be2a81f83728f179194a7/Graphics) contains the graphics of the simulations.

File "[01_MackeyGlass_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/01_MackeyGlass_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the Mackey-Glass dataset.

File "[02_Nonlinear_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/02_Nonlinear_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the Nonlinear dataset.

File "[03_Lorenz_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/03_Lorenz_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the Lorenz Attractor dataset.

File "[NASDAQ_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/NASDAQ_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the NASDAQ dataset.

File "[SP500_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/SP500_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the S&P 500 dataset.

File "[TAIEX_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/TAIEX_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for the TAIEX dataset.

File "[SolarPanelAlice_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/SolarPanelAlice_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for two solar panel datasets from the Alice Springs desert.

File "[SolarPanelYulara_ClassicalSimulations.py](https://github.com/kaikerochaalves/ClassicalModels/blob/cd0dd870b1af03dc915be2a81f83728f179194a7/SolarPanelYulara_ClassicalSimulations.py)" runs the Grid Search for each model and returns the best result for two solar panel datasets from the Yulara.
