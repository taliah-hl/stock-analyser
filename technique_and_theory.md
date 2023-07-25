# Stock Analysis Technique

## 1. Trend
### 1.1 Common Indicators
### 1.1.1 Simple Moving Average (MA)

calculate the average price of a stock over a specified period

Common interval: 20, 50, 200

calculation
```
stock_info = yf.download("NVDA", start="2020-03-01", end="2023-06-30")
stock_data = pd.DataFrame(stock_info["Close"])
#caculate ma20
stock_data['ma20'] = stock_data['Close'].rolling(20).mean()
stock_data['ma20'].dropna(inplace=True)
```


### 1.1.2 Exponential Moving Average (EMA)

Similar to moving averages, but give more weight to recent prices, which can 

provide a more responsive trend estimation

weight more on recent data points by a smoothing factor

calculation

```
# caculate ema20
EMA20 = stock_data['Close'].ewm(span=20, adjust=False).mean()
```

### 1.1.3 Triangular Moving Average (TMA)

reflects a smoothed stock price data over a specific interval, without 


calculate by applying the Simple Moving Average (SMA) twice

calculation

```
# calculate TMA10
ma10 = stock_data['Close'].rolling(10).mean()
tma10 = ma10.rolling(10).mean()
```

### 2. Curve Smoothening
### 2.1 Poly-fit

advantage: no lagging

calculation

```
X = np.array(stock_close_data.reset_index().index)
Y = stock_close_data["Close"].to_numpy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", np.RankWarning)
    poly_fit = np.poly1d(np.polyfit(X, Y, degree))
```

### 2.2 By linear convolution with np.blackman

like smoothen the curve by merging with sine curve

`numpy.blackman(M)`


return: blackman window with M points
`Blackman window` is a taper formed by using the first three terms of a summation of cosines (looks like a cosine peak)

larger M -> smoother peak

Example
`numpy.blackman(5)`

[pic]

`numpy.blackman(10)`

Linear Convolution

- way to convolve a signal with a kernel (like merging 2 signal)
- multiplying corresponding elements of the two signal

calculation with code
```
window = np.blackman(N)
smoothed_data = np.convolve(window / window.sum(), original_data, mode='same')

```
### 2.3 Butterworth Low Pass Filter

ref: https://nehajirafe.medium.com/using-fft-to-analyse-and-cleanse-time-series-data-d0c793bb82e3

- to filter away noise of frequency < certain threshold

transfer function of the Butterworth low-pass filter:

H(z) = (b0 + b1 * z^(-1) + b2 * z^(-2) + ... + bM * z^(-M)) / (1 + a1 * z^(-1) + a2 * z^(-2) + ... + aN * z^(-N))

`scipy.signal.butter` -> find coefficients **a**s and **b**s

`scipy.signal.filtfilt` -> zero-phase filtering is performed to remove the high-frequency noise components from the input data

(expensive function)

## 3. Volitility

measure of how fluctuate is the stock price

### 2.1 Common Indicators

- Bollinger Bands
- Moving Average Convergence Divergence (MACD)
- Average True Range (ATR)


### Reference

numpy.blackman: https://numpy.org/doc/stable/reference/generated/numpy.blackman.html

linear convolution - intuitive explain: https://betterexplained.com/articles/intuitive-convolution/

