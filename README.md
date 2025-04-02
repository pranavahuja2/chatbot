# Basic Chatbot

A simple rule-based chatbot that can handle basic conversations using natural language processing.

## Features

- Basic conversation handling (greetings, goodbyes, thanks)
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Random response selection for more natural-feeling conversations

## Setup

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the chatbot:
```
python chatbot.py
```

The chatbot will start and you can begin chatting. Type 'quit' to exit the conversation.

## How it Works

The chatbot uses NLTK (Natural Language Toolkit) for text processing and implements a simple rule-based response system. It can:
- Recognize and respond to greetings
- Handle goodbyes
- Acknowledge thanks
- Provide default responses for unrecognized inputs

## Limitations

This is a basic implementation that uses pattern matching and predefined responses. It doesn't understand context or maintain conversation history. 