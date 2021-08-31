# Stock

- Find peaks and bottoms
- Find trend

---

- [Stock](#stock)
- [Logic and Design](#logic-and-design)
  - [Peaks and Bottoms](#peaks-and-bottoms)
    - [Example](#example)
    - [Limitations Using Blackman Window](#limitations-using-blackman-window)
  - [Trend](#trend)
    - [Example](#example-1)
    - [Limitations](#limitations)
- [Reference](#reference)
  - [Smoothing the Data ("Noise" Reduction)](#smoothing-the-data-noise-reduction)
  - [Linear Regression](#linear-regression)

---

# Logic and Design

## Peaks and Bottoms

![](./docs/Screenshot%202021-08-16%20184841.png)

- First smooth the stock data and remove the "noise". (See [Smoothing the Data ("Noise" Reduction)](#smoothing-the-data-noise-reduction))
- Find the approximate peaks and bottoms using the smoothed trend
- Find the actual peaks and bottoms using the real data

### Example

![](./docs/NVDA%20Peaks%20and%20Bottoms.png)

### Limitations Using Blackman Window

For a longer period of time, the value of `smooth_data_N` and `find_extrema_interval` has to change to other value.

`"NVDA", start="2010-01-01", end="2021-08-16"`

`smooth_data_N = 15, find_extrema_interval = 5`

![](./docs/NVDA%20Peaks%20and%20Bottoms%202.png)

Solution:

`smooth_data_N = 100, find_extrema_interval = 25`

![](./docs/NVDA%20Peaks%20and%20Bottoms%203.png)

---

## Trend

- Use linear regression and plot a best-fit line.

### Example

`stock_info = yf.download("^HSI", start="2000-01-01", end="2003-06-15")`

![^HSI Linear Regression](./docs/HSI%20Linear%20Regression.png)

`stock_info = yf.download("NVDA", start="2021-01-01", end="2021-08-16")`

![NVDA Linear Regression](./docs/NVDA%20Linear%20Regression.png)

### Limitations

The best-fit line might not be perfect.

`stock_info = yf.download("AAPL", start="2000-01-01", end="2021-08-16")`

![Inaccuracy for long period](./docs/AAPL%20Linear%20Regression.png)

`stock_info = yf.download("AAPL", start="2000-01-01", end="2003-06-15")`

![Sudden drop](./docs/AAPL%20Linear%20Regression%202.png)

---

# Reference

## Smoothing the Data ("Noise" Reduction)

Current approach in smoothing the data (Blackman Window):

https://books.google.com.hk/books?id=m2T9CQAAQBAJ&pg=PA189&lpg=PA189&dq=numpy+blackman+and+convolve&source=bl&ots=5lqrOE_YHL&sig=ACfU3U3onrK4g3uAo3a9FLT_3yMcQXGfKQ&hl=en&sa=X&ved=2ahUKEwjE8p-l-rbyAhVI05QKHfJnAL0Q6AF6BAgQEAM#v=onepage&q=numpy%20blackman%20and%20convolve&f=false

Another approach in smoothing the data with poly fit:

https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

Another approach in smoothing the data:

https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

## Linear Regression

https://medium.com/analytics-vidhya/stock-prediction-using-linear-regression-cd1d8351f536

https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d