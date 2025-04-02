# Import required libraries for natural language processing and API interactions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import re
import sys
import datetime
import requests
from newsapi import NewsApiClient
import spacy
from dateutil import parser
import os
from dotenv import load_dotenv

# Load environment variables from .env file (contains API keys and configuration)
load_dotenv()

def download_nltk_data():
    """
    Downloads required NLTK data packages for natural language processing.
    These packages are essential for text tokenization, stopwords, and lemmatization.
    """
    try:
        nltk.download('punkt', quiet=True)  # For sentence tokenization
        nltk.download('stopwords', quiet=True)  # For filtering common words
        nltk.download('wordnet', quiet=True)  # For lemmatization
        nltk.download('averaged_perceptron_tagger', quiet=True)  # For part-of-speech tagging
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

# Download required NLTK data when the script starts
download_nltk_data()

class NewsChatbot:
    """
    A chatbot class that specializes in delivering news and handling news-related queries.
    Uses spaCy for NLP and NewsAPI for fetching real-time news.
    """
    def __init__(self):
        """
        Initialize the chatbot with required components:
        - spaCy NLP model for text understanding
        - NewsAPI client for fetching news
        - Predefined categories and responses
        """
        try:
            # Load spaCy's English language model for natural language processing
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize News API client with API key from environment variables
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key or api_key == 'your_news_api_key_here':
                print("Warning: Please set your News API key in the .env file")
                self.news_api = None
            else:
                self.news_api = NewsApiClient(api_key=api_key)
            
            # Define available news categories and their display names
            self.categories = {
                'business': 'Business news',
                'entertainment': 'Entertainment news',
                'health': 'Health news',
                'science': 'Science news',
                'sports': 'Sports news',
                'technology': 'Technology news'
            }
            
            # Define response templates for different types of interactions
            self.responses = {
                'greeting': [
                    "Hello! I'm your news assistant. What would you like to know about?",
                    "Hi there! I can help you stay updated with the latest news. What interests you?",
                    "Hey! Ready to explore the latest news? What would you like to know?"
                ],
                'goodbye': [
                    "Goodbye! Stay informed!",
                    "See you later! Keep up with the news!",
                    "Take care! Come back for more news updates!"
                ],
                'thanks': [
                    "You're welcome! Let me know if you need more news updates!",
                    "No problem! Feel free to ask for more news anytime!",
                    "Glad I could help! Stay tuned for more news!"
                ],
                'default': [
                    "I'm not sure about that. Would you like to know about the latest news instead?",
                    "I'm focused on delivering news. Would you like to know about current events?",
                    "I can help you with the latest news. What would you like to know?"
                ]
            }
        except Exception as e:
            print(f"Error initializing components: {e}")
            print("Please make sure all required packages are installed.")
            sys.exit(1)

    def get_news(self, query=None, category=None, count=5):
        """
        Fetch news articles based on query, category, or default to top headlines.
        
        Args:
            query (str): Search query for specific topics
            category (str): News category (business, technology, etc.)
            count (int): Number of articles to return (default: 5)
            
        Returns:
            str: Formatted string containing news articles with details
        """
        if not self.news_api:
            return "News API is not configured. Please set your News API key in the .env file."
            
        try:
            # Fetch news based on the provided parameters
            if query:
                news = self.news_api.get_everything(q=query, language='en', sort_by='publishedAt')
            elif category:
                news = self.news_api.get_top_headlines(category=category, language='en', country='us')
            else:
                news = self.news_api.get_top_headlines(language='en', country='us')
            
            # Process and format the articles
            articles = news['articles'][:count]
            response = "📰 Here are the latest news articles:\n\n"
            
            # Format each article with details
            for i, article in enumerate(articles, 1):
                # Convert ISO date to readable format
                date = parser.parse(article['publishedAt']).strftime("%B %d, %Y")
                
                # Build article response with emojis for better readability
                response += f"📌 {article['title']}\n"
                response += f"   📅 {date}\n"
                response += f"   📰 Source: {article['source']['name']}\n"
                
                # Add description if available
                if article.get('description'):
                    response += f"   📝 {article['description']}\n"
                
                response += f"   🔗 {article['url']}\n\n"
            
            return response
        except Exception as e:
            return f"Sorry, I couldn't fetch the news at the moment. Error: {str(e)}"

    def process_query(self, user_input):
        """
        Process user input and determine appropriate response.
        
        Args:
            user_input (str): The user's input text
            
        Returns:
            str: Appropriate response based on the input
        """
        # Convert input to lowercase for case-insensitive matching
        input_lower = user_input.lower()
        
        # Process input with spaCy for better understanding
        doc = self.nlp(user_input)
        
        # Check for news-related queries
        if any(word in input_lower for word in ['news', 'headlines', 'latest', 'updates']):
            # Try to find a specific category in the query
            found_category = next((cat for cat in self.categories.keys() if cat in input_lower), None)
            
            # Extract potential topic using named entities (organizations, locations, people)
            topic = None
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                    topic = ent.text
                    break
            
            # Return appropriate news based on category or topic
            if found_category:
                return self.get_news(category=found_category)
            elif topic:
                return self.get_news(query=topic)
            else:
                return self.get_news()
        
        # Check for direct category queries
        for category in self.categories.keys():
            if category in input_lower:
                return self.get_news(category=category)
        
        # Handle basic conversation patterns
        if any(word in input_lower for word in ['hello', 'hi', 'hey']):
            return random.choice(self.responses['greeting'])
        
        if any(word in input_lower for word in ['bye', 'goodbye', 'see you']):
            return random.choice(self.responses['goodbye'])
        
        if any(word in input_lower for word in ['thanks', 'thank you', 'appreciate']):
            return random.choice(self.responses['thanks'])
        
        # Default response for unrecognized queries
        return random.choice(self.responses['default'])

def main():
    """
    Main function to run the chatbot.
    Handles the main interaction loop and error handling.
    """
    try:
        # Initialize the chatbot
        chatbot = NewsChatbot()
        
        # Display welcome message and available features
        print("📰 News Chatbot: Hello! I'm your news assistant. I can help you with:")
        print("- Latest news and headlines")
        print("- Category-specific news (business, technology, sports, etc.)")
        print("- News about specific topics, companies, or people")
        print("\nAvailable categories:")
        for category, description in chatbot.categories.items():
            print(f"- {description}")
        print("\nType 'quit' to exit.")
        
        # Main interaction loop
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("News Chatbot: Goodbye! Stay informed!")
                break
            
            response = chatbot.process_query(user_input)
            print("\nNews Chatbot:", response)
            
    except KeyboardInterrupt:
        print("\nNews Chatbot: Goodbye! Stay informed!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Entry point of the script
if __name__ == "__main__":
    main() 