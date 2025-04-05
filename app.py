from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from pyecharts.charts import Line
from pyecharts import options as opts
import os
from datetime import datetime

app = Flask(__name__)

# === 訪問人次計數功能 ===
counter_file = "counter.txt"

def read_counter():
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            return int(f.read())
    return 0

def write_counter(count):
    with open(counter_file, "w") as f:
        f.write(str(count))

# === 工具函式 ===
def preprocess_ticker(ticker):
    ticker = ticker.strip()
    if '.' not in ticker and ticker.isdigit():
        return ticker + '.TW'
    return ticker

def get_close_series(df):
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()

def calculate_metrics(data, bench_data):
    stock_prices = get_close_series(data)
    bench_prices = get_close_series(bench_data)
    total_return = (stock_prices.iloc[-1] - stock_prices.iloc[0]) / stock_prices.iloc[0]
    T = (data.index[-1] - data.index[0]).days / 365.0
    annual_return = (1 + total_return) ** (1 / T) - 1 if T > 0 else np.nan
    stock_ret = stock_prices.pct_change().dropna()
    bench_ret = bench_prices.pct_change().dropna()
    common_index = stock_ret.index.intersection(bench_ret.index)
    if common_index.empty:
        beta = np.nan
    else:
        covariance = stock_ret.loc[common_index].cov(bench_ret.loc[common_index])
        variance = bench_ret.loc[common_index].var()
        beta = covariance / variance if variance != 0 else np.nan
    return total_return, annual_return, beta

# === 主頁路由 ===
@app.route('/', methods=['GET', 'POST'])
def index():
    count = read_counter() + 1
    write_counter(count)

    results = {}
    price_chart_embed, pct_chart_embed = None, None
    tickers_input, start_date, end_date, currency, selected_period = '', '', '', 'TWD', ''

    if request.method == 'POST':
        tickers_input = request.form.get('tickers', '')
        selected_period = request.form.get('period', '')
        currency = request.form.get('currency', 'TWD')

        if selected_period and selected_period != 'custom':
            end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
            periods = {
                '1d': 1, '5d': 5, '1m': 30, '6m': 180,
                'ytd': pd.Timestamp.today().dayofyear,
                '1y': 365, '3y': 365*3, '5y': 365*5,
                '10y': 365*10, '20y': 365*20, 'max': 365*100
            }
            start_date = (pd.Timestamp.today() - pd.Timedelta(days=periods[selected_period])).strftime('%Y-%m-%d')
        else:
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')

        tickers_list = [preprocess_ticker(ticker) for ticker in tickers_input.split(',') if ticker.strip()]

        conversion_rate_series = yf.download("TWDUSD=X", period="1d")['Close']
        conversion_rate = float(conversion_rate_series.iloc[0])
        if conversion_rate < 1:
            conversion_rate = 1 / conversion_rate

        all_data, bench_cache = {}, {}
        for ticker in tickers_list:
            market = 'tw' if ticker.endswith('.TW') else 'us'
            bench_ticker = '^TWII' if market == 'tw' else '^GSPC'
            data = yf.download(ticker, start=start_date, end=pd.to_datetime(end_date)+pd.Timedelta(days=1), auto_adjust=True)
            if data.empty:
                continue
            factor = conversion_rate if market == 'us' and currency == 'TWD' else \
                     1 / conversion_rate if market == 'tw' and currency == 'USD' else 1
            data['Close'] *= factor
            all_data[ticker] = {'data': data, 'bench': bench_ticker}

            if bench_ticker not in bench_cache:
                bench_data = yf.download(bench_ticker, start=start_date, end=pd.to_datetime(end_date)+pd.Timedelta(days=1), auto_adjust=True)
                bench_factor = conversion_rate if bench_ticker == '^GSPC' and currency == 'TWD' else \
                               1 / conversion_rate if bench_ticker == '^TWII' and currency == 'USD' else 1
                bench_data['Close'] *= bench_factor
                bench_cache[bench_ticker] = bench_data

        all_dates = [item['data'].index for item in all_data.values()]
        if not all_dates:
            return render_template('index.html', results={}, price_chart_embed=None, pct_chart_embed=None,
                                   tickers_input=tickers_input, start_date=start_date, end_date=end_date,
                                   selected_period=selected_period, currency=currency,
                                   visitor_count=count, current_year=datetime.now().year)

        combined_dates = sorted(set.union(*map(set, all_dates)))
        for ticker in all_data:
            all_data[ticker]['data'] = all_data[ticker]['data'].reindex(combined_dates).ffill().bfill()
        for bench in bench_cache:
            bench_cache[bench] = bench_cache[bench].reindex(combined_dates).ffill().bfill()

        xaxis = [d.strftime('%Y-%m-%d') for d in combined_dates]

        price_line = Line().add_xaxis(xaxis)
        for ticker, item in all_data.items():
            close_series = item['data']['Close']
            y_values = [round(val.item() if hasattr(val, "item") else float(val), 2) for val in close_series.values]
            price_line.add_yaxis(ticker, y_values, is_smooth=True)
            tot, ann, beta = calculate_metrics(item['data'], bench_cache[item['bench']])
            results[ticker] = {'total_return': tot, 'annual_return': ann, 'beta': beta}
        price_line.set_global_opts(
            title_opts=opts.TitleOpts(title=f"股票價格走勢圖 (幣別：{'台幣' if currency=='TWD' else '美金'})"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts()],
            xaxis_opts=opts.AxisOpts(type_="category")
        )
        price_chart_embed = price_line.render_embed()

        pct_line = Line().add_xaxis(xaxis)
        for ticker, item in all_data.items():
            close_series = item['data']['Close']
            first_val = close_series.iloc[0]
            pct_series = (close_series / first_val - 1) * 100
            y_pct = [round(val.item() if hasattr(val, "item") else float(val), 2) for val in pct_series.values]
            pct_line.add_yaxis(ticker, y_pct, is_smooth=True)
        pct_line.set_global_opts(
            title_opts=opts.TitleOpts(title="股票漲跌幅百分比圖"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[opts.DataZoomOpts()],
            xaxis_opts=opts.AxisOpts(type_="category")
        )
        pct_chart_embed = pct_line.render_embed()

    return render_template('index.html', results=results, price_chart_embed=price_chart_embed,
                           pct_chart_embed=pct_chart_embed, tickers_input=tickers_input,
                           start_date=start_date, end_date=end_date, selected_period=selected_period,
                           currency=currency, visitor_count=count, current_year=datetime.now().year)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
