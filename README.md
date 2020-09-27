<h1 align="center">Machine Price Prediction</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Predicting the price of construction machines using XGBregressor.
</p>
<br> 

## 🧐 About <a name = "about"></a>
Task, as given:
1. Lag en enkel statistikk for noen utvalgte variabler, både visuelt og numerisk. (Snitt, Max, Min).
2. Gjør en vurdering av mengden manglende verdier, konsekvens av dette, og nevn noen metoder for
å håndtere det.
3. Gjør en vurdering om datatyper, kontinuerlige/kategoriske variable, og hvordan dette håndteres.
4. Gjør et modellvalg, og utform en kort begrunnelse som belyser fordeler og ulemper.
5. Velg og beskriv en metric.
6. Beskriv resultatene, samt hvilke features er viktigst.
7. Lag en beskrivelse av hvilke data du valgte som testsett og hvilke data du valgte som treningssett.
8. Redegjør for hvordan du har tatt høyde for overfitting og hvor god overførbarhet modellen har til
nye data.
9. Lag en presentasjon på ca. 8-16 slides med besvarelser på disse oppgavene.
10. Bonusoppgave: Eksponer modellen gjennom et REST API og demonstrer funksjonaliteten til dette
API’et

## 🏁 Getting Started <a name = "getting_started"></a>

All requirements are listed in the 'requirements.txt'-file, simply run the following commands:

```
sudo apt-get install python3.7
sudo apt-get install python3-pip
git clone https://github.com/Sandbergo/machine-price-prediction.git
cd machine-price-prediction
. ./install.sh
cd machine-price-predicition
python3 predict.py
```

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

https://www.ai4love.xyz/2020/05/10/end-to-end-bulldozer-price-regression.html#Add-datetime-parameters-for-saledate-column