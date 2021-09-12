[![Color fonts](https://see.fontimg.com/api/renderfont4/GO6zm/eyJyIjoiZHciLCJoIjoxODIsInciOjIwMDAsImZzIjo5MSwiZmdjIjoiIzIxQkNDNSIsImJnYyI6IiNGRkZGRkYifQ/T1BUSSBmb2xpbw/fattern.png)](https://www.fontspace.com/category/color)

--------------------------------------

![PyPI Version](https://img.shields.io/pypi/v/optifolio)
![License](https://img.shields.io/pypi/l/optifolio)

**OptiFolio** is a package for portfolio optimization. For optimization, a SciPy optimizer is used, while results can be visualized with Bokeh plots.

The package can also be seamlessly integrated with Yahoo Finance API,  using Pandas Data Reader.


Install
-------

OptiFolio can be installed from
[PyPI](https://pypi.org/project/optifolio/):

``` {.sourceCode .python}
pip install optifolio
```


Features
--------

-   **PortfolioOptimizer [object]:** Optimize your portfolio based on Sharpe Ratio.
    * **fit [method]:** Fits daily stock data into the optimizer. Generates annual measures.
    * **plot_efficient_frontier [method]:** Generates a plot for efficient frontier, optimal portfolio, and individual stocks.
    * **plot_weights [method]:** Creates a pie chart that displays portfolio weights for each ticker.
    * **plot_cumulative_return [method]:** Generates a time series plot that displays portfolio performance over time.

The figures are generated with Bokeh, enabling easy implementation to modern web browsers.


Quick Start
-----------

### 1. Optimize Portfolio

````python
from optifolio import PortfolioOptimizer

# Data from Yahoo Finance with Pandas Data Reader
import pandas_datareader.data as web
data = web.DataReader(['AMZN', 'AAPL', 'MSFT',
                       'NFLX', 'TSLA', 'BABA', 'JD'],
                       'yahoo',
                       start='2015/01/01',
                       end='2019/12/31')['Adj Close']

# Initiate the optimizer
model = PortfolioOptimizer()

# Optimize (w. max Sharpe Ratio)
model.fit(data, obj='sharpe')
````

### 2. Visualize Frontier

````python
model.plot_efficient_frontier()
````
<p align="center">
  <img src="https://github.com/kristianbonnici/optifolio/blob/master/img/plot1.png?raw=true" width="800" />
</p>


### 3. Visualize Portfolio Weights

````python
model.plot_weights()
````
<p align="center">
  <img src="https://github.com/kristianbonnici/optifolio/blob/master/img/plot2.png?raw=true" width="800" />
</p>


### 4. Visualize Cumulative Return

````python
# Adding a benchmark to compare against
benchmark = web.DataReader(['^GSPC'],
                           'yahoo',
                           start='2015/01/01',
                           end='2019/12/31')['Adj Close']


model.plot_cumulative_return(benchmark_data=benchmark)
````
<p align="center">
  <img src="https://github.com/kristianbonnici/optifolio/blob/master/img/plot3.png?raw=true" width="800" />
</p>


Author
------

**Kristian Bonnici**

- [Profile](https://github.com/kristianbonnici)
- [Email](mailto:kristian.bonnici@aalto.fi)
- [Website](https://kristianbonnici.github.io/)


🤝 Support
----------

Contributions, issues, and feature requests are welcome!

Give a ⭐️ if you like this project!
