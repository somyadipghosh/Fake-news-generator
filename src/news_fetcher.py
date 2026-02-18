"""
News Fetcher - Fetch real news articles from NewsAPI
"""
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NEWS_API_CONFIG


class NewsAPIFetcher:
    """Fetch real news articles from NewsAPI"""
    
    def __init__(self, api_key=None):
        """
        Initialize NewsAPI fetcher
        
        Args:
            api_key: NewsAPI key (optional, uses config if not provided)
        """
        self.api_key = api_key or NEWS_API_CONFIG['api_key']
        self.base_url = NEWS_API_CONFIG['base_url']
        self.headers = {
            'X-Api-Key': self.api_key
        }
        
    def fetch_top_headlines(self, category=None, sources=None, 
                           country='us', page_size=100, page=1):
        """
        Fetch top headlines
        
        Args:
            category: News category (business, technology, science, health, etc.)
            sources: Comma-separated string of news sources
            country: Country code (us, gb, etc.)
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            List of article dictionaries
        """
        endpoint = f"{self.base_url}/top-headlines"
        
        params = {
            'pageSize': page_size,
            'page': page,
            'language': NEWS_API_CONFIG['language']
        }
        
        if category:
            params['category'] = category
        if sources:
            params['sources'] = sources
        elif country:
            params['country'] = country
            
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['articles']
            else:
                print(f"API error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching headlines: {e}")
            return []
    
    def fetch_everything(self, query=None, sources=None, domains=None,
                        from_date=None, to_date=None, language='en',
                        sort_by='publishedAt', page_size=100, page=1):
        """
        Fetch articles using the /everything endpoint
        
        Args:
            query: Keywords or phrases to search for
            sources: Comma-separated string of news sources
            domains: Comma-separated string of domains
            from_date: Date string (YYYY-MM-DD) or datetime object
            to_date: Date string (YYYY-MM-DD) or datetime object
            language: Language code
            sort_by: Sort order (relevancy, popularity, publishedAt)
            page_size: Number of articles per page (max 100)
            page: Page number
            
        Returns:
            List of article dictionaries
        """
        endpoint = f"{self.base_url}/everything"
        
        params = {
            'pageSize': page_size,
            'page': page,
            'language': language,
            'sortBy': sort_by
        }
        
        if query:
            params['q'] = query
        if sources:
            params['sources'] = sources
        if domains:
            params['domains'] = domains
        if from_date:
            if isinstance(from_date, datetime):
                params['from'] = from_date.strftime('%Y-%m-%d')
            else:
                params['from'] = from_date
        if to_date:
            if isinstance(to_date, datetime):
                params['to'] = to_date.strftime('%Y-%m-%d')
            else:
                params['to'] = to_date
                
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['articles']
            else:
                print(f"API error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching articles: {e}")
            return []
    
    def fetch_sources(self, category=None, language='en', country=None):
        """
        Fetch available news sources
        
        Args:
            category: News category
            language: Language code
            country: Country code
            
        Returns:
            List of source dictionaries
        """
        endpoint = f"{self.base_url}/sources"
        
        params = {}
        if category:
            params['category'] = category
        if language:
            params['language'] = language
        if country:
            params['country'] = country
            
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['sources']
            else:
                print(f"API error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching sources: {e}")
            return []
    
    def process_articles(self, articles):
        """
        Process articles into a standardized format
        
        Args:
            articles: List of article dictionaries from API
            
        Returns:
            DataFrame with processed articles
        """
        processed = []
        
        for article in articles:
            # Skip articles without content
            if not article.get('content') and not article.get('description'):
                continue
                
            # Combine description and content
            text = ''
            if article.get('description'):
                text += article['description'] + ' '
            if article.get('content'):
                # Remove the [+X chars] suffix that NewsAPI adds
                content = article['content']
                content = content.split('[+')[0].strip()
                text += content
            
            # Skip very short articles
            if len(text) < 100:
                continue
            
            processed.append({
                'headline': article.get('title', ''),
                'text': text.strip(),
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'label': 0  # 0 = real news
            })
        
        return pd.DataFrame(processed)
    
    def fetch_diverse_news(self, num_articles=200, delay=1):
        """
        Fetch diverse real news articles from multiple categories and sources
        
        Args:
            num_articles: Target number of articles to fetch
            delay: Delay between API calls (seconds) to avoid rate limits
            
        Returns:
            DataFrame with fetched articles
        """
        print(f"\n{'='*70}")
        print(f"{'FETCHING REAL NEWS FROM NEWSAPI':^70}")
        print(f"{'='*70}\n")
        
        all_articles = []
        categories = NEWS_API_CONFIG['categories']
        articles_per_category = num_articles // len(categories)
        
        for category in categories:
            print(f"\n📰 Fetching {category} news...")
            
            articles = self.fetch_top_headlines(
                category=category,
                country=NEWS_API_CONFIG['country'],
                page_size=min(100, articles_per_category)
            )
            
            all_articles.extend(articles)
            print(f"   ✓ Fetched {len(articles)} articles")
            
            time.sleep(delay)
        
        # Also fetch from reputable sources
        print(f"\n📰 Fetching from trusted sources...")
        sources = ','.join(NEWS_API_CONFIG['sources'][:5])  # First 5 sources
        
        articles = self.fetch_top_headlines(
            sources=sources,
            page_size=50
        )
        
        all_articles.extend(articles)
        print(f"   ✓ Fetched {len(articles)} articles")
        
        # Process articles
        print(f"\n⚙️  Processing articles...")
        df = self.process_articles(all_articles)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        duplicates_removed = initial_count - len(df)
        
        print(f"   ✓ Processed {len(df)} unique articles")
        if duplicates_removed > 0:
            print(f"   ℹ️  Removed {duplicates_removed} duplicates")
        
        return df


def main():
    """Test the news fetcher"""
    print("Testing NewsAPI Fetcher...")
    
    fetcher = NewsAPIFetcher()
    
    # Test fetching headlines
    print("\n1. Fetching top technology headlines...")
    articles = fetcher.fetch_top_headlines(category='technology', page_size=5)
    print(f"   Fetched {len(articles)} articles")
    
    if articles:
        print("\n   Sample article:")
        print(f"   Title: {articles[0].get('title')}")
        print(f"   Source: {articles[0].get('source', {}).get('name')}")
    
    # Test fetching diverse news
    print("\n2. Fetching diverse news...")
    df = fetcher.fetch_diverse_news(num_articles=50, delay=0.5)
    print(f"\n   Total articles fetched: {len(df)}")
    print(f"\n   Sample data:")
    print(df[['headline', 'source']].head())


if __name__ == "__main__":
    main()
