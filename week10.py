from ast import With

from selenium import webdriver
from selenium.webdriver.common.by import By
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# Function to scrape headlines from a given source
def scrape_source(driver, url, selector):
    driver.get(url)
    time.sleep(3)  # Wait for the page to load
    elements = driver.find_elements(By.CSS_SELECTOR, selector)
    headlines = [element.text for element in elements]
    return headlines
# Function to scrape multiple financial news sources
def get_all_sources():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    driver = webdriver.Chrome(options=options)
    headlines = []
    try:
        # Scrape headlines from different financial news sites using CSS seclectors
        headlines += scrape_source(driver, 'https://www.cnbc.com/finance/', '.Card-title,span[data-testid="TitleHeading"]')
        headlines += scrape_source(driver, 'https://www.reuters.com/finance', 'h2,h3')
        headlines += scrape_source(driver, 'https://finance.yahoo.com/', "h2[data-testid='title'],h3")
        headlines += scrape_source(driver, 'https://www.bloomberg.com/markets', 'h3.story-title,a.story-link')
        headlines += scrape_source(driver, 'https://www.marketwatch.com/', 'h3.article__headline,h2.article__headline')
        headlines += scrape_source(driver, 'https://www.wsj.com/news/markets', 'span.WSJTheme--headlineText,h3.WSJTheme--headlineText')
    finally:
        driver.quit()
    return headlines

# Visualize regression model and sentiment scores
def visualize_results(sentiment_scores, model):
    plt.figure(figsize=(10, 7))
    # training data
    x = np.array([0.5, -0.5, 0.4, -0.4, 0.2]).reshape(-1, 1)  # Sample sentiment scores for training
    y = np.array([5, 1, 4, 2, 3])  # Sample stock price changes for training
    plt.scatter(x, y, color='blue', label='Training Data')
    plt.plot(x, model.predict(x), color='black', label='Regression Line')

    # Plot combined sentiment 
    if sentiment_scores is not None:
        prediction = model.predict(np.array([sentiment_scores]).reshape(-1, 1))[0]
        plt.scatter(sentiment_scores, prediction, color='red', s=120, label='Combined Sentiment ')
        plt.text(sentiment_scores, prediction+0.1, "Market Mood", fontsize=10)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Predicted Stock Price Movement')
    plt.title('Combined Market Sentiment vs Stock Movement')
    plt.legend()
    plt.grid(True)
    plt.show()
# Plot histogram of sentiment scores
def plot_histogram(scores):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label=f'Average: {np.mean(scores):.2f}')
    plt.xlabel("Sentiment Score (Compound)")
    plt.ylabel("Number of Headlines" )
    plt.title("Distribution of Headline Sentiment Scores")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    headlines = get_all_sources()
    analyzer = SentimentIntensityAnalyzer()
    
    headlines_data = []
    for h in headlines:
        scores = analyzer.polarity_scores(h)['compound']
        headlines_data.append((h, scores))

with open('scraped_headlines.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Headline", "Sentiment Score"])
    writer.writerows(headlines_data)
print("Scraping and sentiment analysis complete. Data saved to scraped_headlines.csv")

scores = [row[1] for row in headlines_data]
avg_sentiment = np.mean(scores) if scores else None

pos = sum(1 for s in scores if s > 0.05)
neu = sum(1 for s in scores if -0.05 <= s <= 0.05)
neg = sum(1 for s in scores if s < -0.05)

print(f"Positive headlines: {pos}, Neutral headlines: {neu}, Negative headlines: {neg}")
print(f"Combined Average Sentiment: {avg_sentiment:.2f}" if avg_sentiment is not None else "No headlines found.")

X = np.array([0.5, -0.5, 0.4, -0.4, 0.2]).reshape(-1, 1)  # Sample sentiment scores for training
y = np.array([5, 1, 4, 2, 3])  # Sample stock price changes for training
model = LinearRegression()
model.fit(X, y)

visualize_results(avg_sentiment, model)
if scores:
    plot_histogram(scores)