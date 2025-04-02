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

# Load environment variables
load_dotenv()

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

# Download required NLTK data
download_nltk_data()

class EnhancedChatbot:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize News API client
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key or api_key == 'your_news_api_key_here':
                print("Warning: Please set your News API key in the .env file")
                self.news_api = None
            else:
                self.news_api = NewsApiClient(api_key=api_key)
            
            # Basic response patterns
            self.responses = {
                'greeting': [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! How may I assist you?"
                ],
                'goodbye': [
                    "Goodbye! Have a great day!",
                    "See you later!",
                    "Take care!"
                ],
                'thanks': [
                    "You're welcome!",
                    "No problem!",
                    "Glad I could help!"
                ],
                'default': [
                    "I'm not sure I understand. Could you rephrase that?",
                    "I'm still learning. Could you explain that differently?",
                    "I don't have enough information to answer that."
                ]
            }
        except Exception as e:
            print(f"Error initializing components: {e}")
            print("Please make sure all required packages are installed.")
            sys.exit(1)

    def get_current_time(self):
        now = datetime.datetime.now()
        return now.strftime("%I:%M %p")

    def get_current_date(self):
        now = datetime.datetime.now()
        return now.strftime("%B %d, %Y")

    def get_news(self, query=None, category=None):
        if not self.news_api:
            return "News API is not configured. Please set your News API key in the .env file."
            
        try:
            if query:
                news = self.news_api.get_everything(q=query, language='en', sort_by='publishedAt')
            elif category:
                news = self.news_api.get_top_headlines(category=category, language='en', country='us')
            else:
                news = self.news_api.get_top_headlines(language='en', country='us')
            
            articles = news['articles'][:5]  # Get top 5 articles
            response = "Here are the latest news articles:\n\n"
            
            for i, article in enumerate(articles, 1):
                response += f"{i}. {article['title']}\n"
                response += f"   Source: {article['source']['name']}\n"
                response += f"   URL: {article['url']}\n\n"
            
            return response
        except Exception as e:
            return f"Sorry, I couldn't fetch the news at the moment. Error: {str(e)}"

    def process_query(self, user_input):
        # Convert to lowercase for easier matching
        input_lower = user_input.lower()
        
        # Process with spaCy for better NLP understanding
        doc = self.nlp(user_input)
        
        # Check for time-related queries
        if any(word in input_lower for word in ['time', 'clock']):
            return f"The current time is {self.get_current_time()}"
        
        # Check for date-related queries
        if any(word in input_lower for word in ['date', 'day', 'today']):
            return f"Today's date is {self.get_current_date()}"
        
        # Check for news-related queries
        if any(word in input_lower for word in ['news', 'headlines', 'latest']):
            # Extract potential category or topic
            categories = ['business', 'entertainment', 'health', 'science', 'sports', 'technology']
            found_category = next((cat for cat in categories if cat in input_lower), None)
            
            # Extract potential topic using named entities
            topic = None
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                    topic = ent.text
                    break
            
            if found_category:
                return self.get_news(category=found_category)
            elif topic:
                return self.get_news(query=topic)
            else:
                return self.get_news()
        
        # Check for greetings
        if any(word in input_lower for word in ['hello', 'hi', 'hey']):
            return random.choice(self.responses['greeting'])
        
        # Check for goodbyes
        if any(word in input_lower for word in ['bye', 'goodbye', 'see you']):
            return random.choice(self.responses['goodbye'])
        
        # Check for thanks
        if any(word in input_lower for word in ['thanks', 'thank you', 'appreciate']):
            return random.choice(self.responses['thanks'])
        
        # Default response
        return random.choice(self.responses['default'])

def main():
    try:
        chatbot = EnhancedChatbot()
        print("Chatbot: Hello! I'm an enhanced chatbot. I can help you with:")
        print("- Current time and date")
        print("- Latest news and headlines")
        print("- Basic conversation")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            response = chatbot.process_query(user_input)
            print("\nChatbot:", response)
            
    except KeyboardInterrupt:
        print("\nChatbot: Goodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 