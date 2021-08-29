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
        if ret_type == 'log':
            daily_ret = np.log(data / data.shift(1))
        elif ret_type == 'arithmetic':
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
            if self.min_ret < min(self.stock_ret):
                self.min_ret = min(self.stock_ret)
            self.frontier_ret = np.linspace(
                self.min_ret, max(daily_ret.mean() * 252), 30)
        else:
            raise ValueError(
                """The provided input value for min_ret '{}' is over the maximum attainable return.
                Please provide a min_ret that is less than {}""".format(self.min_ret, max(self.stock_ret))
            )

        # ========== init weights (equal distribution) ==========
        init_guess = np.array([1 / (len(data.columns))
                               for _ in range(0, len(data.columns))])

        # ========== weights bounds ==========
        bounds = [(0, 1) for _ in range(0, len(data.columns))]

        # ========== optimize sharpe ratio ==========
        if obj == 'sharpe':
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

                if verbosity == 1:
                    print("Optimize: {}/30 \n Success: {}\n".format(i
                                                                    + 1, opt_results.success))

                self.frontier_vol.append(opt_results['fun'])
                self.frontier_sharpe.append(
                    (ret - self.rf_ret) / opt_results['fun'])
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
        vol = np.sqrt(np.dot(weights.T, np.dot(
            self.daily_ret.cov() * 252, weights)))
        sr = (ret - self.rf_ret) / vol
        return np.array([ret, vol, sr])

    def _min_volatility(self, weights):
        return self._get_return_volatility_sharpe(weights)[1]

    def plot_efficient_frontier(self, width=800, height=500, output="frontier.html", toolbar=False,
                                background_fill_color='#28283B', border_fill_color='#1F1E2C',
                                grid_line_color='#7A7F9B', text_line_color='#F3F6FF'
                                ):

        from bokeh.plotting import ColumnDataSource, figure, output_file, show

        # output to static HTML file
        if output is not None:
            output_file(output)

        tt = [
            ("Name", "@desc"),
            ("Weight", "@size %"),
            ("(Vol, Ret)", "(@x, @y)"),
            ("Sharpe", "@sharpe"),
        ]

        # ========== init figure ==========
        p = figure(plot_width=width, plot_height=height, tooltips=tt,
                   y_range=(min(self.stock_ret)-0.1, max(self.stock_ret)+0.1),
                   x_range=(min(self.frontier_vol)-0.1, max(self.frontier_vol)+0.1),
                   tools="pan,wheel_zoom,save")

        # background & border color
        p.background_fill_color = background_fill_color
        p.border_fill_color = border_fill_color
        # grid
        p.ygrid.grid_line_alpha = 0.5
        p.ygrid.grid_line_dash = [6, 4]
        p.ygrid.grid_line_color = grid_line_color
        p.xgrid.grid_line_alpha = 0.5
        p.xgrid.grid_line_dash = [6, 4]
        p.xgrid.grid_line_color = grid_line_color
        # axis
        p.xaxis.axis_line_color = text_line_color
        p.yaxis.axis_line_color = text_line_color

        p.xaxis.major_label_text_color = text_line_color
        p.yaxis.major_label_text_color = text_line_color

        p.xaxis.major_tick_line_color = text_line_color
        p.xaxis.major_tick_line_width = 3
        p.xaxis.minor_tick_line_color = text_line_color
        p.yaxis.major_tick_line_color = text_line_color
        p.yaxis.major_tick_line_width = 3
        p.yaxis.minor_tick_line_color = text_line_color

        p.xaxis.axis_label = "Risk [Volatility]"
        p.xaxis.axis_label_text_color = text_line_color
        p.xaxis.axis_label_standoff = 20
        p.yaxis.axis_label = "Expected Return"
        p.yaxis.axis_label_text_color = text_line_color
        p.yaxis.axis_label_standoff = 20
        # no logo
        p.toolbar.logo = None
        # toolbar
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
        p.line('x', 'y', line_width=3,
               line_color='#2EE0DD', source=frontier_source)

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

        # ========== individual stocks ==========
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
        if output is not None:
            show(p)

        return p

    def plot_weights(self,
                     width=500,
                     height=500,
                     output="weights.html",
                     background_fill_color='#28283B',
                     border_fill_color='#1F1E2C',
                     colorpalette=None
                     ):
        from math import pi
        from bokeh.io import output_file, show
        from bokeh.plotting import figure
        from bokeh.transform import cumsum

        if colorpalette is None:
            colorpalette = ['#6EDDDC', '#7B67CE', '#55A5E3', '#0088C4', '#8081CB',
                            '#9E5B9D', '#FF7576', '#F0568F', '#BD4AA6', '#A33660',
                            '#83F0BB', '#A4F6A1', '#CCF987', '#FFCC5B', '#F9F871']

        if output is not None:
            output_file(output)

        data = pd.DataFrame(data={'stock': self.stock_names, 'weight': self.stock_weights * 100})
        data['angle'] = data['weight'] / data['weight'].sum() * 2 * pi

        if len(colorpalette) < data.shape[0]:
            while len(colorpalette) < data.shape[0]:
                colorpalette *= 2
        data['color'] = colorpalette[: data.shape[0]]

        p = figure(plot_width=width, plot_height=height, toolbar_location=None,
                   tools="hover", tooltips="@stock: @weight{0.0} %", x_range=(-0.5, 1.0))

        p.wedge(x=0, y=1, radius=0.4,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend_field='stock', source=data)

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        # background & border color
        p.background_fill_color = background_fill_color
        p.border_fill_color = border_fill_color

        if output is not None:
            show(p)

        return p

    def plot_cumulative_return(self, width=800, height=500,
                               benchmark_data=None,
                               output="cumulative_return.html",
                               toolbar=False,
                               background_fill_color='#28283B',
                               border_fill_color='#1F1E2C',
                               grid_line_color='#7A7F9B',
                               text_line_color='#F3F6FF'
                               ):

        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, RangeTool
        from bokeh.io import output_file, show
        from bokeh.plotting import figure

        weighted_norm_ret = {}
        for col in self.data:
            weighted_norm_ret[col] = self.data[col] / self.data.iloc[0][col]
        weighted_norm_ret = pd.DataFrame(weighted_norm_ret)
        for col, weight in zip(weighted_norm_ret, self.stock_weights):
            weighted_norm_ret[col] = weighted_norm_ret[col] * weight
        weighted_norm_ret['Total'] = weighted_norm_ret.sum(axis=1)

        source = ColumnDataSource(data=dict(date=weighted_norm_ret.index, close=weighted_norm_ret['Total']))

        p = figure(plot_width=width, plot_height=height, x_axis_type="datetime",
                   x_range=(min(weighted_norm_ret.index), max(weighted_norm_ret.index)),
                   tools="pan,wheel_zoom,save")
        p.grid.grid_line_alpha = 0.9

        p.line('date', 'close', source=source, color='#2EE0DD', legend_label='Portfolio', line_width=1.5)

        if benchmark_data is not None:
            benchmark_data.iloc[0:] = benchmark_data.iloc[0:] / benchmark_data.iloc[0]
            source2 = ColumnDataSource(data=dict(date=benchmark_data.index, close=benchmark_data.iloc[:, 0]))
            p.line('date', 'close', source=source2, color='#D544B1',
                   legend_label=benchmark_data.columns[0], line_width=1.5)

        p.legend.location = "top_left"

        # background & border color
        p.background_fill_color = background_fill_color
        p.border_fill_color = border_fill_color
        # grid
        p.ygrid.grid_line_alpha = 0.5
        p.ygrid.grid_line_dash = [6, 4]
        p.ygrid.grid_line_color = grid_line_color
        p.xgrid.grid_line_alpha = 0.5
        p.xgrid.grid_line_dash = [6, 4]
        p.xgrid.grid_line_color = grid_line_color
        # axis
        p.xaxis.axis_line_color = text_line_color
        p.yaxis.axis_line_color = text_line_color
        p.xaxis.major_label_text_color = text_line_color
        p.yaxis.major_label_text_color = text_line_color
        p.xaxis.major_tick_line_color = text_line_color
        p.xaxis.major_tick_line_width = 3
        p.xaxis.minor_tick_line_color = text_line_color
        p.yaxis.major_tick_line_color = text_line_color
        p.yaxis.major_tick_line_width = 3
        p.yaxis.minor_tick_line_color = text_line_color
        p.yaxis.axis_label = "Return"
        p.yaxis.axis_label_text_color = text_line_color
        p.yaxis.axis_label_standoff = 20
        # no logo
        p.toolbar.logo = None
        # toolbar
        if toolbar is False:
            p.toolbar_location = None

        select = figure(plot_height=height, plot_width=width, y_range=p.y_range,
                        x_axis_type="datetime", y_axis_type=None,
                        tools="", toolbar_location=None, background_fill_color="#efefef")

        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "#D544B1"
        range_tool.overlay.fill_alpha = 0.1

        select.line('date', 'close', source=source, color='#2EE0DD')
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        select.toolbar.active_multi = range_tool

        # background & border color
        select.background_fill_color = background_fill_color
        select.border_fill_color = border_fill_color
        # grid
        select.ygrid.grid_line_alpha = 0.5
        select.ygrid.grid_line_dash = [6, 4]
        select.ygrid.grid_line_color = grid_line_color
        select.xgrid.grid_line_alpha = 0.5
        select.xgrid.grid_line_dash = [6, 4]
        select.xgrid.grid_line_color = grid_line_color
        # axis
        select.xaxis.axis_line_color = text_line_color
        select.yaxis.axis_line_color = text_line_color
        select.xaxis.major_label_text_color = text_line_color
        select.yaxis.major_label_text_color = text_line_color
        select.xaxis.major_tick_line_color = text_line_color
        select.xaxis.major_tick_line_width = 3
        select.xaxis.minor_tick_line_color = text_line_color
        select.yaxis.major_tick_line_color = text_line_color
        select.yaxis.major_tick_line_width = 3
        select.yaxis.minor_tick_line_color = text_line_color
        # no logo
        select.toolbar.logo = None

        if output is not None:
            output_file("cumulative_return.html", title="cumulative return")
            show(column(p, select))

        return column(p, select)
