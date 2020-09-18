<h1 align="center">Predicting Molecular Properties</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Prediction of the scalar coupling constant of atom pairs in organic molecules from tabular data using ensembling of gradient boosting trees (XGB) and deep neural networks (DNN) methods in a separate model based meta-architecture. The project used data from the Kaggle competition champs-scalar-coupling.
</p>
<br> 

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
Molecular representation with distance matrices and additional generated angle data used for accurate predictions of a quantum mechanical property. XGB and DNNs were found to have comparable accuracy (with XGB generally better) and ensembling these methods with a strongly separated configuration gave satisfactory results. This repository contains all code needed to replicate our results and can be modified for different methods or datasets.

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Prerequisites
All requirements are listed in the 'requirements.txt'-file, simply run the following commands:

```
sudo apt-get install python3.7
sudo apt-get install python3-pip
git clone https://github.com/teamtoll/predicting-molecular-properties.git
cd predicting-molecular-properties
python -m pip install -r requirements.txt
```

Kaggle API setup: https://github.com/Kaggle/kaggle-api.

### Installing

Kaggle Download:

Downloads and extracts all necessary data source files from the Kaggle competition and organizes it into a data_sources directory, ready to use.

```
cd utils
python kaggle_download.py
```
Follow any instructions given as output in case of missing files or directories.

Generated files can be downloaded from (place within ./input): https://drive.google.com/file/d/1JN35qpWmMxRAXO28XfLr42ALx1w0Gcia/view?usp=sharing

### File Structure

The hierarchy should look like this:

    .
    ├── input                         
    │     ├── features
    |     |    └── ...
    │     ├── generated
    |     |    └── ...
    │     └── zipped_source
    |          └── ...
    ├── models                         
    │     ├── nn
    |     │    ├── nn_model_1JHC.hdf5
    |     |    └── ...
    │     └── xgb
    |          ├── xgb_model_1JHC.hdf5
    |          └── ...
    ├── notebooks                              
    │     └── main.ipynb
    ├── submissions                         
    │     └── submission_best.csv
    ├── utils                         
    │     ├── other        
    |     |    ├── distance_matrix.py
    |     |    └── ...
    │     ├── check_repository.py
    │     └── ...
    |
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt


## 🎈 Usage <a name="usage"></a>
Run the notebook notebooks/main.iypnb, tweak hyper-parameters, change up the data, see where it goes.
This repository can also be used as a basis for a completely different problem and dataset. 

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
- [Jupyter Notebook](https://jupyter.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/) 
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)
- Fredrik Bakken [@FredrikBakken](https://github.com/FredrikBakken) 

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Hallvar Gisnås [@hallvagi](https://github.com/hallvagi)
- Lars Aurdal [@larsaurdal](https://github.com/larsaurdal)
- Dennis Christensen [@dennis-christensen](https://github.com/dennis-christensen)
- Niels Aase
- Kyle Lobo [@kylelobo](https://github.com/kylelobo)
