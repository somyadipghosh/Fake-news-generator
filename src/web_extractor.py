"""
Web Article Extractor - Extract content from article URLs
"""
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse


class ArticleExtractor:
    """Extract article content from URLs"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_from_url(self, url, timeout=10):
        """
        Extract article content from URL
        
        Args:
            url: Article URL
            timeout: Request timeout in seconds
            
        Returns:
            dict with 'title', 'text', 'url', 'success', 'error'
        """
        result = {
            'success': False,
            'url': url,
            'title': '',
            'text': '',
            'error': None
        }
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                result['error'] = "Invalid URL format"
                return result
            
            # Fetch page
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract article text
            text = self._extract_article_text(soup)
            
            if not text or len(text.strip()) < 100:
                result['error'] = "Could not extract meaningful content from the page"
                return result
            
            result['success'] = True
            result['title'] = title
            result['text'] = text
            
        except requests.exceptions.Timeout:
            result['error'] = f"Request timeout after {timeout} seconds"
        except requests.exceptions.ConnectionError:
            result['error'] = "Could not connect to the URL"
        except requests.exceptions.HTTPError as e:
            result['error'] = f"HTTP error: {e}"
        except Exception as e:
            result['error'] = f"Error extracting article: {str(e)}"
        
        return result
    
    def _is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_title(self, soup):
        """Extract article title from HTML"""
        # Try different title sources
        title = None
        
        # Try <title> tag
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        # Try og:title meta tag
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content'].strip()
        
        # Try h1 tag
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()
        
        # Try article title class
        if not title:
            article_title = soup.find(['h1', 'h2'], class_=re.compile(r'title|headline', re.I))
            if article_title:
                title = article_title.get_text().strip()
        
        return title or "Untitled"
    
    def _extract_article_text(self, soup):
        """Extract main article text from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                            'aside', 'form', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find article content
        article_text = None
        
        # Try <article> tag
        article = soup.find('article')
        if article:
            article_text = self._clean_text(article.get_text())
        
        # Try common article content classes/ids
        if not article_text or len(article_text) < 100:
            for selector in ['article-body', 'article-content', 'post-content', 
                           'entry-content', 'article__body', 'story-body',
                           'content-body', 'article-text']:
                content = soup.find(['div', 'section'], class_=re.compile(selector, re.I))
                if not content:
                    content = soup.find(['div', 'section'], id=re.compile(selector, re.I))
                if content:
                    text = self._clean_text(content.get_text())
                    if len(text) > len(article_text or ''):
                        article_text = text
        
        # Try all paragraphs as fallback
        if not article_text or len(article_text) < 100:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([self._clean_text(p.get_text()) for p in paragraphs])
        
        # Final fallback - get all text
        if not article_text or len(article_text) < 100:
            article_text = self._clean_text(soup.get_text())
        
        return article_text
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text


def extract_article_from_url(url, timeout=10):
    """
    Convenience function to extract article from URL
    
    Args:
        url: Article URL
        timeout: Request timeout
        
    Returns:
        dict with article data
    """
    extractor = ArticleExtractor()
    return extractor.extract_from_url(url, timeout)
