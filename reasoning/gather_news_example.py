import requests
import pandas as pd
from datetime import datetime

# Replace 'API_KEY' with your actual API key from NewsAPI
api_key = 'API ENV'

# Categories to fetch news for
categories = ['business', 'general', 'technology']

# Prepare today's date in the format 'YYYY-MM-DD'
today = datetime.today().strftime('%Y-%m-%d')

# List to store all articles
articles_list = []


# Loop through each category and fetch news
for category in categories:
    url = f'https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={api_key}'
    
    # Make the request to the NewsAPI
    response = requests.get(url)
    2
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        
        # Process each article and append to articles_list
        for article in articles:
            title = article['title']
            description = article['description']
            content = article['content']
            article_url = article['url']
            published_at = article['publishedAt']
            
            # Convert the published date to 'YYYY-MM-DD' format
            published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d')
            
            # Append article details to list
            articles_list.append({
                'Date': published_date,
                'Category': category,
                'Headline': title,
                'Description': description,
                'Content': content if content else 'No content available',
                'URL': article_url
            })
    else:
        print(f"Failed to retrieve news for category: {category}")

# Create a DataFrame from the list of articles
df = pd.DataFrame(articles_list)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv(f'news_data/news_{today}.csv', index=False)
