# How to run

## Historical data

### 1, Install the dataset from [here](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

### 2. Unzip it and place it in 'data' folder

### 3. Run the prepare.py

### 3.1. Alternatively if you want to use just a few stocks instead of the whole dataset run limited_prepare.py and select the stocks you want

### 4. After that run main.py

### 5. Run time_sequences.py

## Economic data

### 1. Install datasets from these links (you need to change the key)

#### https://www.alphavantage.co/query?function=WTI&interval=monthly&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={key}&datatype=csv

#### https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey={key}&datatype=csv

#### https://fred.stlouisfed.org/series/PCU2122212122210

### 2. Place it in 'economic_data' folder

### 3. Run economic_data.py
