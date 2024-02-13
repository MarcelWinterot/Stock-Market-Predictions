# Training plans for the model

## 1. Training the model on the kaggle dataset

### This step will involve processing and cleaning the data

### After that a neural network will be trained on it to get the best results possible

#### More on this later

## 2. RL play time

### After the model has a satisfying accuracy we will move onto reinforcment learning

### I will create or find a stock market enviroment where the model can interact and invest in real time

#### I will most likely use one of the following:

- https://github.com/AminHP/gym-anytrading

- https://gym-trading-env.readthedocs.io/en/latest/#

- https://github.com/AminHP/gym-mtsim

## 3. Robinhood play time

### If the model succeeds in generating a profit I will let it play for a while on Robinhood

## Plans for the model

### 1. Historical data analysis

#### The first part of the model will be analyzing the last year of stocks of a company

#### It will probably be somekind of RNN or a transformer

#### It should also have time2vec

### 2. Sentiment analysis of the news

#### The second part will be a sentiment analysis model of the news regarding the company or the world at the moment

#### I will be using [this model](https://huggingface.co/ProsusAI/finbert)

### 3. Economical analysis

#### Another part of the model will look at current economics of the world, the country of the company and strategic and luxury resources

#### It will probably have simillar desing the the historical data analysis but we will see

### 4. MLP

#### To combine all of the parts of the model I will create a small MLP

#### Don't know how it will look like yet but probably a FNN
