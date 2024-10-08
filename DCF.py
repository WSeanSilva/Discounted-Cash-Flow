import warnings
import math
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader as pdr
from rich.console import Console
from rich.table import Table

console = Console()

# surpressing future warnings
warnings.filterwarnings("ignore")


# Function to get the stock ticker input
def get_ticker_input():
    ticker = input("Enter the stock ticker of choice: ")
    return ticker


# Function to calculate cost of equity using Fama-French 3-Factor model
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

    h3Beta = res3.params['Mkt-RF']
    ErSMB = data['SMB'].mean()
    ErHML = data['HML'].mean()
    Bhml = res3.params['HML']
    Bsmb = res3.params['SMB']
    Rf = data['RF'].mean()
    Emrp = data['Mkt-RF'].mean()

    f3COE = Rf + (h3Beta * Emrp) + (Bsmb * ErSMB) + (Bhml * ErHML)
    return f3COE * 12 * 100



# Function to calculate WACC
def wacc(ticker, return_equity):
    
    stock = yf.Ticker(ticker)
    annual_bs = stock.balance_sheet
    annual_is = stock.financials

    # Market Cap
    total_equity = stock.info['marketCap']
    
    tax_provision = annual_is.loc["Tax Provision"].iloc[0]
    ebt = annual_is.loc["Pretax Income"].iloc[0]

    # Calculate the tax rate
    
    tax_rate = tax_provision / ebt

    # Debt Calculations
    bs_debt = annual_bs.loc["Total Debt"].iloc[0]
    interest_expense = annual_is.loc["Interest Expense"].dropna().iloc[0]
    pre_tax_cd = interest_expense / bs_debt
    cost_debt = pre_tax_cd * (1-tax_rate)
    avg_interest_rate = interest_expense / bs_debt
    
    # Mkt Value of Debt
    
    total_debt = interest_expense * ((1 / ((1 + cost_debt)**avg_interest_rate)) / cost_debt) + (bs_debt / ((1 + cost_debt)**avg_interest_rate))

    # Calculate WACC
    wacc_value = (total_debt / (total_debt + total_equity) * cost_debt) * (1 - tax_rate) + \
                 (total_equity / (total_debt + total_equity) * (return_equity / 100))
    
    return wacc_value * 100, tax_rate, ebt, stock, total_debt, annual_bs  # Multiply by 100 to get percentage



def free_cash_flow(ticker, tax_rate, ebit, stock, total_debt, annual_bs):
	
	annual_cf = stock.cashflow  # corrected to 'cashflow'
	
	# Depreciation
	depreciation = annual_cf.loc["Depreciation Amortization Depletion"].iloc[0]
	
	# CapEx and NWC (fixed typo for Capital Expenditure)
	capX = annual_cf.loc["Capital Expenditure"].iloc[0]
	
	current_assets = annual_bs.loc["Current Assets"].iloc[0]
	current_liabilities = annual_bs.loc["Current Liabilities"].iloc[0]
	changeNWC = current_assets - current_liabilities
	
	# FCFF Calculation
	fcff = ebit * (1 - tax_rate) + depreciation - capX - changeNWC
	
	return fcff


def discounted_cash_flow(ticker, fcff_value, wacc_value):
    
    # Determine the size of the growth period array
    p_vector_size = int(input("Enter how many different growth periods there will be before a terminal value is calculated? : "))
    
    # Create lists for period and growth rate values
    period_vector = [0] * p_vector_size
    growth_vector = [0] * p_vector_size
    
    # Input periods
    for i in range(p_vector_size):
        period_vector[i] = int(input(f"Enter the time-length for growth period #{i+1}: "))
    
    # Input growth rates
    for i in range(p_vector_size):
            growth_vector[i] = float(input(f"Enter the growth rate for period #{i+1} (e.g., 0.05 for 5%): "))
    
    # Terminal growth rate
    terminal_growth = float(input("Enter the terminal growth rate after the last growth period (e.g., 0.03 for 3%): "))
    
    # Calculate the intrinsic value
    intrinsic_value = 0
    current_fcff = fcff_value
    year = 1
    
    # Discount free cash flows for each period
    for i in range(p_vector_size):
        for _ in range(period_vector[i]):
            discounted_fcff = current_fcff / ((1 + wacc_value) ** year)
            intrinsic_value += discounted_fcff
            current_fcff *= (1 + growth_vector[i])
            year += 1
    
    # Calculate terminal value
    terminal_value = (current_fcff * (1 + terminal_growth)) / (wacc_value - terminal_growth)
    discounted_terminal_value = terminal_value / ((1 + wacc_value) ** year)
    
    # Add terminal value to intrinsic value
    intrinsic_value += discounted_terminal_value
    
    # Get the number of shares outstanding from yfinance
    stock = yf.Ticker(ticker)
    shares = stock.info.get('sharesOutstanding', None)
    
    # Calculate estimated price per share
    estimated_price = intrinsic_value / shares if shares else None
    
    return intrinsic_value, estimated_price


# Function to display table with rich
def table_display(ticker, wacc_value, return_equity, fcff_value, intrinsic_value, estimated_price):  # add fcff_value as parameter
    
    # Create table
    table = Table(title=f"===Financial Data For {ticker}===", style="bold cyan")
    
    # Column for financial and values
    table.add_column("Financial", justify="left", style="green")
    table.add_column("Calculation", justify="left", style="yellow")
    
    # rows for financial data
    table.add_row("WACC", f"{wacc_value:.2f}%")
    table.add_row("Cost of Equity", f"{return_equity:.2f}%")
    table.add_row("Free Cash Flow To The Firm", f"${fcff_value}") 
    table.add_row("Instrinsic Value", f"${intrinsic_value}")  
    table.add_row("Estimated Price Per Share", f"${estimated_price}")  
    
    # print table
    console.print(table)


def main():
    ticker = get_ticker_input()
    return_equity = cost_equity(ticker)
    
    # Unpack all the values returned by wacc function
    wacc_value, tax_rate, ebit, stock, total_debt, annual_bs = wacc(ticker, return_equity)
    
    # Calculate FCFF
    fcff_value = free_cash_flow(ticker, tax_rate, ebit, stock, total_debt, annual_bs)
    
    # Calculate intrinsic value and estimated price per share using DCF
    intrinsic_value, estimated_price = discounted_cash_flow(ticker, fcff_value, wacc_value)
    
    # Display table with WACC, COE, FCFF, intrinsic value, and estimated price
    table_display(ticker, wacc_value, return_equity, fcff_value, intrinsic_value, estimated_price)
    
if __name__ == "__main__":
    main()
