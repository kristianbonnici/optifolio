"""Main module."""

from scipy.optimize import minimize
import pandas as pd
import numpy as np


# Author: Kristian Bonnici <kristiandaaniel@gmail.fi>


class PortfolioOptimizer:

    def __init__(self):
        self.data = None
        self.daily_ret = None
        self.min_ret = None
        self.rf_ret = None

        # optimal portfolio
        self.sharpe = -999
        self.ret = -999
        self.vol = -999
        self.log_return = None
        self.volatility = None
        self.weights = None
        self.scipy_result_object = None

        # frontier
        self.frontier_vol = []
        self.frontier_ret = []
        self.frontier_sharpe = []

        # individual stocks
        self.stock_names = None
        self.stock_weights = None
        self.stock_vol = None
        self.stock_ret = None
        self.stock_sharpe = None

    def fit(self, data, obj='sharpe', ret_type='log', min_ret=0.03, rf_ret=0.01, verbosity=0):

        # ========== base data ==========
        self.data = data
        self.rf_ret = rf_ret
        self.min_ret = min_ret

        # ========== daily returns ==========
        if ret_type is 'log':
            daily_ret = np.log(data / data.shift(1))
        elif ret_type is 'arithmetic':
            daily_ret = data.pct_change(1)
        else:
            raise ValueError(
                """The provided input value for ret_type '{}' is not supported.
                This input value should be one of the following: {}""".format(ret_type, ['log', 'arithmetic'])
            )
        self.daily_ret = daily_ret

        # ========== stock data ==========
        self.stock_names = data.columns.values
        self.stock_ret = daily_ret.mean() * 252
        self.stock_vol = daily_ret.std() * np.sqrt(252)
        self.stock_sharpe = (self.stock_ret - rf_ret) / self.stock_vol

        # ========== frontier returns ==========
        if self.min_ret < max(self.stock_ret):
            self.frontier_ret = np.linspace(self.min_ret, max(daily_ret.mean() * 252), 30)
        else:
            raise ValueError(
                """The provided input value for min_ret '{}' is over the maximum attainable return.
                Please provide a min_ret that is less than {}""".format(self.min_ret, max(self.stock_ret))
            )

        # ========== init weights (equal distribution) ==========
        init_guess = np.array([1 / (len(data.columns)) for _ in range(0, len(data.columns))])

        # ========== weights bounds ==========
        bounds = [(0, 1) for _ in range(0, len(data.columns))]

        # ========== optimize sharpe ratio ==========
        if obj is 'sharpe':
            for i, ret in enumerate(self.frontier_ret):
                # constraints to equal 0
                cons = ({'type': 'eq', 'fun': self._check_sum},
                        {'type': 'eq', 'fun': lambda w: self._get_return_volatility_sharpe(w)[0] - ret})

                # Sequential Least SQuares Programming (SLSQP).
                opt_results = minimize(fun=self._min_volatility,
                                       x0=init_guess,
                                       method='SLSQP',
                                       bounds=bounds,
                                       constraints=cons)

                if verbosity is 1:
                    print("Optimize: {}/30 \n Success: {}\n".format(i + 1, opt_results.success))

                self.frontier_vol.append(opt_results['fun'])
                self.frontier_sharpe.append((ret - self.rf_ret) / opt_results['fun'])
                if self.frontier_sharpe[-1] > self.sharpe:
                    self.sharpe = self.frontier_sharpe[-1]
                    self.ret = ret
                    self.vol = opt_results['fun']
                    self.scipy_result_object = opt_results

        else:
            raise ValueError(
                """The provided input value for obj '{}' is not supported.
                This input value should be one of the following: {}""".format(obj, ['sharpe'])
            )

        # ========== optimal weights ==========
        self.stock_weights = self.scipy_result_object.x

        # ========== return self ==========
        return self

    def _check_sum(self, weights):
        """
        Returns 0 if sum of weights is 1.0
        """
        return np.sum(weights) - 1

    def _neg_sharpe_ratio(self, weights):
        return self._get_return_volatility_sharpe(weights)[2] * -1

    def _get_return_volatility_sharpe(self, weights):
        weights = np.array(weights)
        ret = np.sum(self.daily_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.daily_ret.cov() * 252, weights)))
        sr = (ret - self.rf_ret) / vol
        return np.array([ret, vol, sr])

    def _min_volatility(self, weights):
        return self._get_return_volatility_sharpe(weights)[1]

    def plot_efficient_frontier(self, width=800, height=500, output="frontier.html", toolbar=False):
        from bokeh.plotting import ColumnDataSource, figure, output_file, show

        # output to static HTML file
        output_file(output)

        tt = [
            ("Name", "@desc"),
            ("Weight", "@size %"),
            ("(Vol, Ret)", "(@x, @y)"),
            ("Sharpe", "@sharpe"),
        ]

        # ========== init figure ==========
        p = figure(plot_width=width, plot_height=height, tooltips=tt)
        # backfround & border color
        p.background_fill_color = '#2A2B4A'
        p.border_fill_color = '#1A1F3C'
        # grid
        p.ygrid.grid_line_alpha = 0.5
        p.ygrid.grid_line_dash = [6, 4]
        p.ygrid.grid_line_color = '#7A7F9B'
        p.xgrid.grid_line_alpha = 0.5
        p.xgrid.grid_line_dash = [6, 4]
        p.xgrid.grid_line_color = '#7A7F9B'
        # axis
        p.xaxis.axis_line_color = "#F3F6FF"
        p.yaxis.axis_line_color = "#F3F6FF"

        p.xaxis.major_label_text_color = "#F3F6FF"
        p.yaxis.major_label_text_color = "#F3F6FF"

        p.xaxis.major_tick_line_color = "#F3F6FF"
        p.xaxis.major_tick_line_width = 3
        p.xaxis.minor_tick_line_color = "#F3F6FF"
        p.yaxis.major_tick_line_color = "#F3F6FF"
        p.yaxis.major_tick_line_width = 3
        p.yaxis.minor_tick_line_color = "#F3F6FF"

        p.xaxis.axis_label = "Risk [Volatility]"
        p.xaxis.axis_label_text_color = "#F3F6FF"
        p.xaxis.axis_label_standoff = 20
        p.yaxis.axis_label = "Expected Return"
        p.yaxis.axis_label_text_color = "#F3F6FF"
        p.yaxis.axis_label_standoff = 20
        # no logo
        p.toolbar.logo = None
        # toolbas
        if toolbar is False:
            p.toolbar_location = None

        # ========== frontier ==========
        frontier_source = ColumnDataSource(data=dict(
            x=self.frontier_vol,
            y=self.frontier_ret,
            desc=len(self.frontier_vol) * ['efficient frontier'],
            size=len(self.frontier_vol) * [0],
            sharpe=self.frontier_sharpe,
        ))
        p.line('x', 'y', line_width=3, line_color='#2EE0DD', source=frontier_source)

        # ========== optimal sharpe ==========
        optim_source = ColumnDataSource(data=dict(
            x=[self.vol],
            y=[self.ret],
            desc=['optimal portfolio'],
            size=[100],
            sharpe=[self.sharpe],
        ))
        p.circle('x', 'y', size=20, fill_color='#45D7B4', fill_alpha=0.8,
                 line_color='#45D7B4', line_width=1.5, source=optim_source)

        # individual stocks
        stocks_source = ColumnDataSource(data=dict(
            x=self.stock_vol,
            y=self.stock_ret,
            desc=self.stock_names,
            size=self.stock_weights * 100,
            sharpe=self.stock_sharpe,
        ))
        p.square('x', 'y', color='#D544B1', fill_alpha=0.2,
                 size='size', source=stocks_source)
        p.scatter('x', 'y', color='#D544B1', source=stocks_source)

        # ========== show results ==========
        show(p)

    def plot_weights(self):
        pass

    def plot_cumulative_return(self):
        pass
