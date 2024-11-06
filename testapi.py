import requests
import yfinance as yf
import pandas as pd
import time

# Set up the API key for your LLM service
api_key = "ljoWc7l91mdURBRNTNPlZQlGpV5OMo"  # Replace with your actual API key

# LLM Service Configuration
llm_url = "http://localhost:8000/chat/completions"
llm_headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

# Initialize portfolio
portfolio = {
    'cash': 10000.0,  # Starting with $10,000
    'positions': {},  # Stock positions
    'transaction_history': []  # Record of transactions
}

# Function to execute trades based on LLM recommendation
def execute_trade(recommendation, current_price, date):
    if recommendation == 'buy':
        # Calculate the number of shares to buy (e.g., invest 10% of cash)
        investment_amount = portfolio['cash'] * 0.1
        shares_to_buy = investment_amount / current_price
        portfolio['cash'] -= investment_amount
        portfolio['positions']['NVDA'] = portfolio['positions'].get('NVDA', 0) + shares_to_buy
        portfolio['transaction_history'].append({
            'date': date,
            'action': 'buy',
            'shares': shares_to_buy,
            'price': current_price,
            'total': investment_amount
        })
        print(f"Bought {shares_to_buy:.2f} shares at ${current_price:.2f} per share on {date}.")
    elif recommendation == 'sell':
        if 'NVDA' in portfolio['positions'] and portfolio['positions']['NVDA'] > 0:
            shares_to_sell = portfolio['positions']['NVDA']
            proceeds = shares_to_sell * current_price
            portfolio['cash'] += proceeds
            portfolio['positions']['NVDA'] = 0
            portfolio['transaction_history'].append({
                'date': date,
                'action': 'sell',
                'shares': shares_to_sell,
                'price': current_price,
                'total': proceeds
            })
            print(f"Sold {shares_to_sell:.2f} shares at ${current_price:.2f} per share on {date}.")
        else:
            print(f"No shares to sell on {date}.")
    else:
        print(f"Holding position on {date}.")

# Function to calculate portfolio value
def calculate_portfolio_value(current_price):
    total_value = portfolio['cash']
    for stock, shares in portfolio['positions'].items():
        total_value += shares * current_price
    return total_value

# Fetch historical stock data for NVDA using yfinance
stock_data = yf.download('NVDA', start='2023-01-01', end='2023-12-31')

# Ensure the data is sorted by date and reset index
stock_data = stock_data.sort_index().reset_index()

# Loop over each day in the stock data
for index, row in stock_data.iterrows():
    # Prepare the prompt with recent data up to the current day
    recent_data = stock_data.loc[max(0, index - 4):index].to_string(index=False)
    date_str = row['Date'].strftime('%Y-%m-%d')
    prompt = f"""
You are a financial analyst. Based on the following recent stock data for Nvidia (NVDA), provide a buy, hold, or sell recommendation. Respond with only one word: buy, hold, or sell.

Stock Data:
{recent_data}

Recommendation:
"""
    # Call the LLM service
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "model": "llama3.1-8b",
        "service_name": "cerebras",
        "temperature": 0.7,
        "max_tokens": 50,
        "top_p": 0.9,
    }
    response = requests.post(llm_url, json=data, headers=llm_headers)
    if response.status_code == 200:
        result = response.json()
        recommendation = result["content"].strip().lower()
        # Ensure the recommendation is valid
        if recommendation not in ['buy', 'hold', 'sell']:
            print(f"Invalid recommendation '{recommendation}' on {date_str}. Skipping trade.")
            continue
        print(f"Date: {date_str}, LLM Recommendation: {recommendation}")
        # Execute the trade
        current_price = row['Close']
        execute_trade(recommendation, current_price, date_str)
        # Optionally, pause or wait for the next time interval
        time.sleep(0.1)  # Small delay to simulate time between trades
    else:
        print(f"Error {response.status_code}: {response.text}")
        break  # Exit the loop on error

# Final portfolio value calculation
final_price = stock_data.iloc[-1]['Close']
final_portfolio_value = calculate_portfolio_value(final_price)
total_return = ((final_portfolio_value - 10000) / 10000) * 100

# Display final portfolio performance
print("\nFinal Portfolio Performance:")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Cash Available: ${portfolio['cash']:.2f}")
print(f"Positions: {portfolio['positions']}")
print("\nTransaction History:")
for transaction in portfolio['transaction_history']:
    print(transaction)