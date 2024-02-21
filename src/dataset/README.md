# How to run

## Historical data

### 1, Install the dataset from [here](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

### 2. Unzip it and place it in 'data' folder

### 3. Run prepare_historical.py

## Economic data

### 1. Install datasets from these links (you need to change the key)

#### https://www.alphavantage.co/query?function=WTI&interval=monthly&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey={key}&datatype=csv

#### https://fred.stlouisfed.org/series/PCU2122212122210

### 2. Place it in 'economic_data' folder

### 3. Run prepare_economic.py

### 4. If you want to use model_3 you need to run combine.py
