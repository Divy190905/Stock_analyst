import yfinance as yf
from collections import Counter
import requests
import praw
from groq import Groq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import math
from datetime import datetime, timedelta
import tensorflow as tf
# Initialize the Groq API client
client = Groq(api_key="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")

# Function to get financial data for a specific stock symbol
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    data = {
        "Current Price": info.get("currentPrice", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Price-to-Book Ratio": info.get("priceToBook", "N/A"),
        "Debt-to-Equity Ratio": info.get("debtToEquity", "N/A"),
        "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
        "Revenue Growth": info.get("revenueGrowth", "N/A"),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Day High": info.get("dayHigh", "N/A"),
        "Day Low": info.get("dayLow", "N/A")
    }
    return data

# Function to fetch news for a specific stock or general market news
def fetch_news(stock_name=None):
    news_api_key = '3a75f8781761465da09403ccbe14a19f'
    news_base_url = 'https://newsapi.org/v2/everything'
    params = {
        'apiKey': news_api_key,
        'language': 'en',
        'pageSize': 5,
        'sortBy': 'publishedAt'
    }
    params['q'] = stock_name or 'Indian Stock Market OR Global Market Factors'
    try:
        response = requests.get(news_base_url, params=params)
        response.raise_for_status()
        news_data = response.json()
        if news_data['totalResults'] == 0 or len(news_data['articles']) < 5:
            params['q'] = 'Indian Stock Market OR Global Market Factors'
            response = requests.get(news_base_url, params=params)
            response.raise_for_status()
            news_data = response.json()
        return [
            f"{article['title']}\n{article['description']}" for article in news_data['articles']
        ]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# Function to send each news article to the GROQ API and get recommendations
def recommend_top_stocks_from_article(article_content):
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f"Based on the following news article, list the top 5 Indian stock names "
                    f"mentioned in the article. Only return the stock names, one per line. "
                    f"If no stock news is present, recommend general stocks:\n\n{article_content}"
                )
            }],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        recommendation = ""
        for chunk in completion:
            recommendation += chunk.choices[0].delta.content or ""
        stocks = [stock.strip() for stock in recommendation.strip().split('\n') if stock.strip()]
        return stocks[:5]
    except Exception as e:
        print(f"Error fetching recommendations from GROQ API: {e}")
        return []

# Function to process and analyze news articles
def process_news_articles(news_articles):
    all_stocks = []
    for article in news_articles:
        recommended_stocks = recommend_top_stocks_from_article(article)
        if not recommended_stocks:
            recommended_stocks = recommend_top_stocks_from_article("General stock recommendations for the Indian market.")
        if len(recommended_stocks) < 2:
            recommended_stocks = list(set(recommended_stocks + ['HDFC', 'TCS', 'Nifty']))
        all_stocks.extend(recommended_stocks)
    return Counter(all_stocks)

# Function to display top stocks
def display_top_stocks(stock_counter):
    filtered_stocks = {stock: count for stock, count in stock_counter.items() if len(stock) <= 10}
    top_5_stocks = sorted(filtered_stocks.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n--- Top Stocks Based on News Articles ---")
    for stock, count in top_5_stocks:
        print(f"{stock}")
# Set up Reddit API credentials
REDDIT_CLIENT_ID = 'IhrERBkIZvPfgwBPaeYSPQ'
REDDIT_SECRET = 'uIbFlDxKHSFGjWp_Ussk3qc1mY0d2Q'
REDDIT_USER_AGENT = 'my_stock_sentiment_bot v1.0'

# Define sector keywords
SECTOR_KEYWORDS = {
    "Energy": ["energy", "oil", "gas", "renewable", "power"],
    "Technology": ["technology", "IT", "software", "hardware", "AI", "cloud"],
    "Finance": ["finance", "bank", "investment", "insurance", "equity"],
    "Healthcare": ["healthcare", "pharmaceutical", "biotech", "hospital", "medical"],
    "Industrials": ["industrial", "manufacturing", "engineering", "construction"],
    "Consumer Goods": ["consumer", "FMCG", "retail", "lifestyle", "fashion"],
    "Utilities": ["utilities", "electric", "water", "energy"],
}

# Sentiment keywords
POSITIVE_KEYWORDS = ["buy", "bullish", "increase", "gain", "profit", "positive", "growth"]
NEGATIVE_KEYWORDS = ["sell", "bearish", "decline", "loss", "negative", "drop", "fall"]

# Function to fetch Reddit posts based on stock name or general Indian stock market
def fetch_reddit_posts(stock_name=None):
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_SECRET,
                         user_agent=REDDIT_USER_AGENT)
    
    subreddit_name = 'IndianStockMarket'
    posts = []

    try:
        search_query = f"{stock_name} stock" if stock_name else '(NSE OR BSE OR "Indian stock market")'
        for submission in reddit.subreddit(subreddit_name).search(search_query, sort='new', limit=10):
            posts.append(submission)

        return posts

    except Exception as e:
        print(f"Error fetching Reddit posts: {e}")
        return None

# Function to extract post details, classify by sector, and detect sentiment
def extract_post_details(posts):
    post_details = []
    for post in posts:
        sentiment = detect_sentiment(post.title, post.selftext)
        post_content = f"{post.title} {post.selftext if post.selftext else 'No description available'}"
        post_info = {
            'content': post_content,
            'sector': classify_sector(post_content),
            'sentiment': sentiment,
            'upvotes': post.score,
            'top_comment': get_top_comment(post)
        }
        if sentiment == "Positive":
            post_details.append(post_info)
    post_details.sort(key=lambda x: x['upvotes'], reverse=True)
    return post_details

# Function to classify post by sector based on keywords
def classify_sector(content):
    combined_text = content.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in combined_text for keyword in keywords):
            return sector
    return "Uncategorized"

# Function to detect sentiment based on keywords
def detect_sentiment(title, description):
    combined_text = f"{title} {description}".lower()
    if any(word in combined_text for word in POSITIVE_KEYWORDS):
        return "Positive"
    elif any(word in combined_text for word in NEGATIVE_KEYWORDS):
        return "Negative"
    return "Neutral"

# Helper function to get the most upvoted comment
def get_top_comment(post):
    post.comments.replace_more(limit=0)
    top_comment = max(post.comments.list(), key=lambda c: c.score, default=None)
    return top_comment.body if top_comment else "No comments"

# GROQ API call to recommend top stocks based on post content
def recommend_top_stocks(post_content):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": f"based on the given post just recommend the top stocks and just give the name of the stock and nothing else not even the top line\n\n{post_content}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Collect and structure the response
    recommendation = ""
    for chunk in completion:
        recommendation += chunk.choices[0].delta.content or ""
    
    # Split the response by lines and store as a list of recommended stocks
    return recommendation.strip().split('\n')
def predict_stock_prices(ticker):
    # Set random seeds for reproducibility
    np.random.seed(42)  # For NumPy
    tf.random.set_seed(42)  # For TensorFlow


    # Calculate the start date (two years before today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1100)

    # Fetch data using yfinance
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # Use only the 'Close' prices for prediction
    df1 = data['Close']

    # Plot the closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(df1)
    plt.title(f'{ticker} Stock Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduce x-axis ticks
    plt.xticks(
        ticks=df1.index[::len(df1)//12],
        labels=pd.to_datetime(df1.index[::len(df1)//12]).strftime("%b '%y"),
        rotation=45
    )
    plt.show()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split the data into training and test sets
    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # Function to create dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    # Prepare the datasets
    time_step = 50
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=40,restore_best_weights=True)
    model.fit(X_train, y_train,validation_data=(X_test, ytest), batch_size=64, epochs=200, callbacks=[early_stopping], verbose=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original scale
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    

    # Calculate RMSE
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))
    
    
    look_back = time_step
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(df1), label='Actual Prices')
    plt.plot(trainPredictPlot, label='Train Predictions')
    plt.plot(testPredictPlot, label='Test Predictions')
    plt.title(f'{ticker} Stock Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(
        ticks=range(0, len(df1), len(df1)//12),
        labels=pd.to_datetime(data.index[::len(df1)//12]).strftime("%b '%y"),
        rotation=45
    )
    plt.legend()
    plt.show()
    
    
    future_steps = 30
    forecast_input = test_data[-time_step:]  # Last `time_step` days of test data
    forecast_input = forecast_input.reshape(1, -1, 1)  # Reshape to match model input
    forecast = []

    for _ in range(future_steps):
        # Predict the next day
        next_value = model.predict(forecast_input, verbose=0)
        forecast.append(next_value[0, 0])  # Store the prediction

        # Update the input with the predicted value (reshape correctly)
        next_value = next_value.reshape(1, 1, 1)  # Reshape next_value to match the input shape (1, 1, 1)
        forecast_input = np.append(forecast_input[:, 1:, :], next_value, axis=1)  # Append the predicted value

# Inverse transform the forecasted values to original scale
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Plot the forecasted prices
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(df1), label='Actual Prices')
    plt.plot(range(len(df1), len(df1) + future_steps), forecast, label='Forecast', color='red')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Customize x-axis for forecast plot
    total_len = len(df1) + future_steps
    xticks_positions = range(0, total_len, len(df1)//12 if len(df1) >= 12 else 1)
    xticks_labels = pd.date_range(
        start=start_date, 
        periods=total_len, 
        freq='D'
    ).strftime("%b '%y")
    plt.xticks(ticks=xticks_positions, labels=xticks_labels[::len(df1)//12], rotation=45)

    plt.legend()
    plt.show()
    
    
def main(stock_name=None):
    if stock_name:
        financial_data = get_stock_data(stock_name)
        print("\n--- Financial Data ---")
        for key, value in financial_data.items():
            print(f"{key}: {value}")
    news_articles = fetch_news()
    if not news_articles:
        print("No news available to process.")
        return
    stock_counter = process_news_articles(news_articles)
    display_top_stocks(stock_counter)
    
    recommend = []
    reddit_posts = fetch_reddit_posts(None)
    post_details = extract_post_details(reddit_posts)
    
    stock_counter = Counter()  # Counter to track frequency of recommended stocks
    
    # Generate recommendations for each post
    for post in post_details:
        top_stocks = recommend_top_stocks(post['content'])
        recommend.append(top_stocks)  # Append the list of recommendations to the recommend array
        
        # Update the stock frequency counter
        stock_counter.update(top_stocks)
    
    print("\nTop  Recommended Stocks From Public Sentiment Across Reddit:")
    for stock, count in stock_counter.most_common(10):
        print(f"{stock}")

    predict_stock_prices(stock_name)

# Run the main function
if __name__ == "__main__":
    stock_name = input("Enter stock name (e.g., 'AAPL' for Apple) or press enter for general market news: ")
    main(stock_name)
