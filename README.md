
<!-- BADGES AREA -->
[![GitHub issues](https://img.shields.io/github/issues/IMsumitkumar/No-code-ML-platform-DashB.ai?style=for-the-badge)](https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai/issues)
[![GitHub forks](https://img.shields.io/github/forks/IMsumitkumar/No-code-ML-platform-DashB.ai?style=for-the-badge)](https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai/network)
[![GitHub stars](https://img.shields.io/github/stars/IMsumitkumar/No-code-ML-platform-DashB.ai?style=for-the-badge)](https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai/stargazers)
[![GitHub license](https://img.shields.io/github/license/IMsumitkumar/No-code-ML-platform-DashB.ai?style=for-the-badge)](https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai)
![GitHub repo size](https://img.shields.io/github/repo-size/IMsumitkumar/No-code-ML-platform-DashB.ai?style=for-the-badge)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">DashB.ai</h3>
  <p align="center">
    <a href="">Video Demo</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Overview](#overview)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Run](#run)
* [preprocessing pipeline tree](#preprocessing-pipeline-tree)
* [Directory Tree](#directory-tree)
* [Contributing](#contributing)
* [Team](#team)
* [License](#license)
* [Contact](#contact)
* [References](#references)
* [Credits](#credits)



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- IMAGE -->
![main page](https://i.imgur.com/9UA6Rkg.png)

### Overview

- This is a web app that automates the data preprocessing pipeline.Target is to automate the whole machine learning pipeline.But this project is final till data preprocessing pipeline.
- Currently this project is in developement phase.
- User can upload comma seperated value files or directly fetch the data from mysql database.(Make sure mysql is installed in your system).
- User's have all the command what to perform and what to not so selected operations can be passed to the pipeline to showcase the result.
- User's can visualize the data using dataviz tool comes along with Dash.ai which can visualize the data without writing any code. (Made by Dash by plotly)


### Built With

[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/64/000000/python.png"/>](https://www.python.org/)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/64/000000/html-5.png"/>](https://www.w3schools.com/html/)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/48/000000/css3.png"/>](https://www.w3schools.com/css/)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/48/000000/javascript.png"/>](https://www.w3schools.com/js/DEFAULT.asp)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/48/000000/linux.png"/>](https://www.linux.org/)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/48/000000/django.png"/>](https://www.djangoproject.com/)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/color/48/000000/sql.png"/>](sqhttps://www.mysql.com/l)
[<img align="left" alt="sumit" width="33px" src="https://img.icons8.com/fluent/48/000000/github.png"/>](https://github.com)
<br>
<br>
- Bootstrap
- scikit learn
- plotly

<br/>
<br/>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.
make sure [git](https://git-scm.com/downloads) is installed in yout machine.

### Installation

1. Clone the repo
```sh
git clone https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai
```
2. create a virtual env and activate
```bash
conda create -n <env_name> python=3.7
conda activate <env_name>
```
2. Install dependencies
```bash
pip install -r requirements.txt      -      (inside project directory)
```

### RUN

> STEP 1 : Migrate the databse tables and create superuser

```python
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser

    username : *****
    email    : *****
    password : ******
```

> STEP 2
```python
python manage.py runserver
```

> STEP 3 : OPTIONAL
For email recovery you have to set our credentials in DashB -> settings.py

```
Set your email and password
```

## Preprocessing Pipeline Tree

```
├── Handle Datatypes
│   ├── Drop unnecessary features.
│   ├── replace inf with NaN.
│   ├── Make sure all the column names are of string type and clean them.
│   ├── Remove the column if target column has NaN.
│   ├── Remove Duplicate columns
│   ├── handle numerical, catergorical and time features.
│   └── Try to determine Ml usecase and encode.
├── Handle Missing Values
│   ├────── Numerical Features
│   ├── Replace with mean.
│   ├── Replace with median.
│   ├── Repalce with Mode.
│   ├── Replace with standard deviation.
│   ├── Replace with zero.
│   ├────── Categorical Features
│   ├── Replace with mean.
│   ├── Replace with "Missing".
│   └── Repalce with Most frequent value.
├── Removing zero and near zero variance columns
│   ├── Eliminate the features that have zero varinace,
│   └── Eliminate the features that have near zero variace.
├── Group Similiar Features
│   └── Group more than two features Make new features with them.
├── Normalization and Transformation
│   ├────── Operations to apply only on numerical features
│   ├── ZScore
│   ├── MinMax
│   ├── Quantile
│   ├── MaxAbs
│   ├── Yeo-Johnson
│   ├────── Target t7ransformation (regression)
│   ├── Box-Cox
│   └── Yeo-Johnson
├── Making Time Features
│   ├── Take a time feature and extract more features from it
│   └── (Day, Month, Year, Hour, Minute, Second, Quantile, Quarter, Day of week, week day name, day of year, week of year )
├── Feature Encoding
│   ├────── Ordinal Encoding
│   ├── LabelEncoding
│   ├── Target Guided ordinal encoding
│   ├────── One hot encoding
│   ├── KDD orange
│   ├── Mean Encoding
│   └── Counter/frequency encoding
├── Removing Outliers
│   ├── Isolaton Forest
│   ├── KNN
│   ├── PCA
│   └── Elliptical envelope
├── Feature Selection
│   ├── Chi squared (Not working perfectly)
│   ├── RFE (Not working on all the data)
│   ├── Lasso (works perfectly)
│   ├── Random Forest
│   ├── lgbm (works perfectly)
│   └── Remove zero variance features
├── Imbalance Dataset (Not done yet)
│   ├── Ensemble techniques automatically handles imblance dataset
│   ├── Undersampling (Not a good idea)
│   ├── Oversampling 
│   ├── SMOTE
│   └── Isolation Forest
└──NExt Step
```




## Directory Tree 
```
├── accounts 
│   └─────────── # handles login, signup and password recovery. 
├── DashB
│   └─────────── # main folder contains wsgi, routing, settings and urls.
├── data
│   └─────────── # main folder for performing pipeline.
├── Viz
│   └─────────── # project app for data visualizatio tool.
├── static
│   └─────────── # contains static files.
├── media
│   └─────────── # storage folder of uploaded media.
├── templates
│   └─────────── # contains landing page templates
├── manage.py
├── requirements.txt
├── LICENSE
├── README.md
└── db.sqlite3

```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Team
[![Sumit Kumar](https://img.icons8.com/color/48/000000/linux.png)]() |
-|
[Sumit](https://github.com/IMSumitKumar) |)


<!-- LICENSE -->
## License
![APM](https://img.shields.io/apm/l/vim-mode?color=blue&style=for-the-badge)

Copyright 2020 Sumit Kumar

<!-- CONTACT -->
## Contact

Sumit Kumar - email me @[sksumit068@gmail.com](https://mail.google.com/)

Project Link: [https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai](https://github.com/IMsumitkumar/No-code-ML-platform-DashB.ai)

## References

- https://docs.djangoproject.com/en/3.1/
- https://www.djangoproject.com/
- https://www.youtube.com/channel/UCTZRcDjjkVajGL6wd76UnGg
- https://plotly.com/
- https://pycaret.org/
- https://scikit-learn.org/
- https://getbootstrap.com/docs/4.0/getting-started/introduction/
- https://django-plotly-dash.readthedocs.io/en/latest/
- https://www.kaggle.com/
- https://www.researchgate.net/publication/220320826_Winning_the_KDD_Cup_Orange_Challenge_with_Ensemble_Selection


## Credits
- HTML templates are being used from open source.
- Modificatons are made by me.