# covid19-survive-predictor

> A COVID-19 survival machine learning predictor using the dataset from Kaggle to tell you whether you will be live or dead based on your gender and age.

I only managed to find a dataset that fit in my simple model at [Kaggle](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset). The dataset only update until February 2020.

<p align="left">
  <img src="assets/screenshot.png" width="auto" height="auto">
</p>

### _setup

```bash
$ git clone https://github.com/loouislow81/covid19-survive-predictor.git
$ cd covid19-survive-predictor
```

set up a new python3 virtual environment for this repo,

```bash
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt --default-timeout=1000
```

### _usage

```bash
$ (venv) python predict.py
```

---

[MIT](https://github.com/loouislow81/covid19-survive-predictor/blob/master/LICENSE)
