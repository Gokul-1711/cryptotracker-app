import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pycoingecko import CoinGeckoAPI
from openai import OpenAI
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize API clients
cg = CoinGeckoAPI()
openai_client = OpenAI(api_key=openai_api_key)

# Function to get cryptocurrency data
@st.cache_data(ttl=300)
def get_crypto_data():
    coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=250, page=1, sparkline=False)
    return pd.DataFrame(coins)

# Function to get historical data
@st.cache_data(ttl=3600)
def get_historical_data(coin_id, days):
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to get GPT-4 analysis
def get_gpt4_analysis(coin_data, historical_data):
    prompt = f"""Analyze the following cryptocurrency data:

Coin: {coin_data['name']}
Current Price: ${coin_data['current_price']}
Market Cap: ${coin_data['market_cap']:,}
24h Volume: ${coin_data['total_volume']:,}
24h Change: {coin_data['price_change_percentage_24h']}%

Historical price data (last 30 days):
{historical_data[['date', 'price']].tail(30).to_string()}

Based on this data, provide a brief analysis of the cryptocurrency's performance, potential trends, and factors that might be influencing its price. Keep the analysis concise but informative."""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a cryptocurrency analyst providing insights based on market data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Function to get GPT-4 advice
def get_gpt4_advice(question, crypto_data):
    top_10_cryptos = crypto_data.head(10)[['name', 'current_price', 'market_cap', 'price_change_percentage_24h']].to_string()

    prompt = f"""You are a cryptocurrency advisor. Use the following data about the top 10 cryptocurrencies by market cap to help answer the user's question. If the question is not directly related to this data, use your general knowledge about cryptocurrencies to provide a helpful and informative answer.

Top 10 Cryptocurrencies:
{top_10_cryptos}

User's question: {question}

Please provide a concise, informative, and helpful answer."""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable cryptocurrency advisor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("Cryptocurrency Tracker and Advisor")

    # Fetch crypto data
    df = get_crypto_data()

    # User input for coin selection
    user_input = st.text_input("Enter the name of the cryptocurrency you want to check:", "Bitcoin")

    # Find the closest match to user input
    df['lower_name'] = df['name'].str.lower()
    user_input_lower = user_input.lower()
    closest_match = df[df['lower_name'].str.contains(user_input_lower)].iloc[0] if any(df['lower_name'].str.contains(user_input_lower)) else None

    if closest_match is not None:
        coin_data = closest_match
        st.success(f"Showing data for {coin_data['name']}")

        # Display current information
        st.header(f"{coin_data['name']} Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${coin_data['current_price']:,.2f}")
        col2.metric("Market Cap", f"${coin_data['market_cap']:,.0f}")
        col3.metric("24h Volume", f"${coin_data['total_volume']:,.0f}")
        col4.metric("24h Change", f"{coin_data['price_change_percentage_24h']:.2f}%")

        # Historical data and chart
        st.header("Historical Data and Trends")
        days = st.slider("Select number of days", 1, 365, 30)
        historical_data = get_historical_data(coin_data['id'], days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data['date'], y=historical_data['price'], mode='lines', name='Price'))
        fig.update_layout(title=f"{coin_data['name']} Price Over Time", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)

        # GPT-4 Analysis
        st.header("AI-Powered Market Analysis")
        if st.button("Generate Analysis"):
            with st.spinner("Analyzing market data..."):
                analysis = get_gpt4_analysis(coin_data, historical_data)
            st.write(analysis)
    else:
        st.error("Cryptocurrency not found. Please check the spelling or try another name.")

    # Cryptocurrency Advisor
    st.header("Cryptocurrency Advisor")
    user_question = st.text_input("Ask a question about cryptocurrencies:")
    if user_question:
        with st.spinner("Generating advice..."):
            advice = get_gpt4_advice(user_question, df)
        st.write(advice)

if __name__ == "__main__":
    main()
