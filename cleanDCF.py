import warnings
import math
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader as pdr


def get_ticker_input():
    ticker = input("Enter the stock ticker of choice: ")
    return ticker



def cost_equity(ticker):
    ff3f = pdr.DataReader('F-F_Research_Data_Factors', 'famafrench', '1950-01-01')[0] / 100

    # Download monthly prices (keep only Adjusted Close prices)
    firm_prices = yf.download(ticker, '2000-12-01', '2024-07-01', interval='1mo')['Adj Close'].dropna().to_frame()

    # Calculate monthly returns, drop missing, convert from Series to DataFrame
    firm_ret = firm_prices.pct_change().dropna()

    # Rename "Adj Close" to the ticker
    firm_ret.rename(columns={'Adj Close': ticker}, inplace=True)

    # Convert index to monthly period date
    firm_ret.index = firm_ret.index.to_period('M')

    # Merge the two datasets
    data = firm_ret.join(ff3f)
    data['const'] = 1

    # Set up the data
    y = data[ticker] - data['RF']
    X3 = data[['const', 'Mkt-RF', 'SMB', 'HML']]

    # Run regression
    res3 = sm.OLS(y, X3).fit()
	
	# Setting up equation
    h3Beta = res3.params['Mkt-RF']
    ErSMB = data['SMB'].mean()
    ErHML = data['HML'].mean()
    Bhml = res3.params['HML']
    Bsmb = res3.params['SMB']
    Rf = data['RF'].mean()
    Emrp = data['Mkt-RF'].mean()

    COE = Rf + (h3Beta * Emrp) + (Bsmb * ErSMB) + (Bhml * ErHML)
    return COE * 12 * 100
    



def financials(ticker):
	
	# Getting financial data
	stock = yf.Ticker(ticker)
	annual_bs = stock.balance_sheet
	annual_is = stock.financials
	annual_cf = stock.cashflow
	
	# Financials for WACC	
	tax_provision = annual_is.loc["Tax Provision"].iloc[0]
	ebt = annual_is.loc["Pretax Income"].iloc[0]	
	tax_rate = tax_provision / ebt
	bs_debt = annual_bs.loc["Total Debt"].iloc[0]
	interest_expense = annual_is.loc["Interest Expense"].dropna().iloc[0]
	pre_tax_cd = interest_expense / bs_debt
	cost_debt = pre_tax_cd * (1-tax_rate)
	avg_interest_rate = interest_expense / bs_debt
	ebit = annual_is.loc["EBIT"].iloc[0]
	
# ------------------------------------------------------------------------ #
	
	# WACC Calculation
	wacc = (total_debt / (total_debt + total_equity) * cost_debt) * (1 - tax_rate) + \
	(total_equity / (total_debt + total_equity) * (return_equity / 100))
	
	# Financials for FCFF
	depreciation = annual_cf.loc["Depreciation Amortization Depletion"].iloc[0]
	
	capX = annual_cf.loc["Capital Expenditure"].iloc[0]
	
	current_assets = annual_bs.loc["Current Assets"].iloc[0]
	current_liabilities = annual_bs.loc["Current Liabilities"].iloc[0]
	dNWC = current_assets - current_liabilities
	
	# FCFF Calculation
	fcff = ebit * (1 - tax_rate) + depreciation - capX - dNWC
# ------------------------------------------------------------------------ #

	# Expected Growth Rate
	reinvestment_rate = (capX - depreciation + dNWC) / (ebit * (1 - tax_rate))
	roc = (ebit + interest_expense)
	
	print(fcff)

	return wacc, fcff

def main():
	get_ticker_input(ticker)
	cost_equity()
	financials()

main()
