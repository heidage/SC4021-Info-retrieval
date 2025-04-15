import praw
import pandas as pd
import yaml
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load configuration from ../config.yaml
def load_config(config_file='../config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load Reddit API credentials and scraping config from config.yaml
config = load_config()
CLIENT_ID = config['reddit']['client_id']
CLIENT_SECRET = config['reddit']['client_secret']
USER_AGENT = config['reddit']['user_agent']

subreddits = config['scraping']['subreddits']
posts_per_subreddit = config['scraping']['posts_per_subreddit']

# Initialize Reddit instance
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# List of expanded keywords related to trading platforms and user experiences
keywords = [    
    # Fees and costs
    "fee", "commission", "pricing", "cost", "expensive", "cheap", "affordable", "subscription",
    "monthly fee", "annual fee", "transaction fee", "hidden charges", "price", "pricing plan",
    
    # Trading features
    "options", "stock", "stocks", "futures", "forex", "crypto", "etf", "mutual fund", 
    "bonds", "derivatives", "margin", "leverage", "short selling", "fractional shares",
    "pre-market", "after-hours", "extended hours", "limit order", "stop loss", 
    
    # User experience
    "interface", "dashboard", "ui", "ux", "user interface", "intuitive", "clunky", "buggy",
    "responsive", "lag", "glitch", "crash", "latency", "loading time", "navigation",
    
    # Features and tools
    "chart", "charting", "analysis", "research", "screener", "scanner", "alert", "notification",
    "watchlist", "portfolio", "tracking", "news feed", "indicator", "technical analysis",
    "fundamental analysis", "educational", "tutorial", "mobile app", "desktop",
    
    # Support and service
    "customer service", "support", "help", "response time", "customer care", "assistance",
    "live chat", "phone support", "email support", "onboarding", "account opening",
    
    # Opinion indicators
    "experience", "review", "thoughts", "feedback", "opinions", "recommendation", "suggest",
    "avoid", "stay away", "prefer", "better than", "worse than", "compared to", "versus",
    "good", "bad", "terrible", "excellent", "amazing", "awful", "horrible", "great",
    "recommend", "worth it", "not worth", "disappointed", "satisfied", "impressed", 
    "frustrating", "reliable", "unreliable", "convenient", "inconvenient", "user-friendly",
    "complicated", "straightforward", "easy to use", "difficult", "learning curve",
    
    # Evaluation terms
    "pros", "cons", "advantages", "disadvantages", "like", "dislike", "hate", "love",
    "rating", "star", "rank", "compare", "comparison", "alternative", "switch",
    "best", "worst", "top", "overrated", "underrated", "value for money"
]

# Function to check if the comment is opinionated based on sentiment
def is_opinionated(comment_body):
    sentiment = analyzer.polarity_scores(comment_body)
    return sentiment['compound'] > 0.3 or sentiment['compound'] < -0.3  

# Scrape comments from subreddit
def scrape_subreddit(subreddit_name, limit=posts_per_subreddit):
    # Get the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    
    # Collect posts and comments data
    comments_data = []
    for submission in subreddit.hot(limit=limit):
        # Collect post details
        submission.comments.replace_more(limit=0)  # Remove "MoreComments"
        
        for comment in submission.comments.list():
            # Filter out blank comments and comments that are too short
            if not comment.body.strip() or len(comment.body.split()) < 5:
                continue            
            if not any(keyword.lower() in comment.body.lower() for keyword in keywords):
                continue
            if not is_opinionated(comment.body):  # Only consider opinionated comments
                continue
            
            # Collect relevant comment details
            created_timestamp = datetime.utcfromtimestamp(comment.created_utc)
            readable_date = created_timestamp.strftime("%d %B %Y")  

            # Collect comment information
            comment_info = {
                'subreddit': subreddit_name,
                'title': submission.title,
                'post_url': submission.url,
                'comment_body': comment.body,
                'comment_author': comment.author.name if comment.author else 'unknown',
                'created': comment.created_utc,
                'readable_date': readable_date  
            }
            comments_data.append(comment_info)
    
    return comments_data

# Save data to CSV
def save_to_csv(data, filename="results.csv"):
    # Save the data to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Main function to scrape comments from all subreddits
def main():
    all_comments = []
    
    for subreddit in subreddits:
        print(f"Scraping comments from subreddit: {subreddit}")
        comments = scrape_subreddit(subreddit, limit=posts_per_subreddit)
        all_comments.extend(comments)
    
    print("Saving data to results.csv...")
    save_to_csv(all_comments)

if __name__ == "__main__":
    main()
