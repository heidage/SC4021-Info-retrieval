import os
import time
import logging
import praw
import prawcore
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

from SP500DataLoader import SP500DataLoader

load_dotenv()

# Configure logging
logging.basicConfig(
    filename="reddit_scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting Reddit Scraper....")

# Initialize PRAW with credentials
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

# Subreddits to scrape
# subreddits = ["StocksAndTrading", "StockMarketMovers"]
subreddits = ["stocks", "stockstobuytoday", "StockMarket", "wallstreetbets"]
# current_count = 9177
target = 10000 # Total posts to collect
posts_per_subreddit = 500
sorting_methods = ["hot", "new", "top"]



# Load stock tickers
stock_tickers = SP500DataLoader().get_ticker_list()
logging.info(f"Loaded {len(stock_tickers)} stock tickers.")

# Storage
data = []
seen_post_ids = set()
# read in csv file to get the current count of posts
# df = pd.read_csv("reddit_scraper_2025-04-04_05-00-32.csv")
# seen_post_ids = set(df['post_id'])

def contains_stock_mention(text):
    """Check if a post/comment contains any stock ticker."""
    if not text:
        return False
    text = text.lower()
    return any(ticker.lower() in text for ticker in stock_tickers)

# Loop to scrape
for subreddit_name in subreddits:
    # total_count = df.shape[0]
    total_count = 0
    for sort in sorting_methods:
        logging.info(f"Scraping subreddit {subreddit_name} with {sort} sorting method.")

        retries = 0  # Track retry attempts
        max_retries = 5  # Max retries before skipping

        while retries < max_retries:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                posts = list(getattr(subreddit, sort)(limit=posts_per_subreddit))
                
                logging.info(f"Found {len(posts)} posts in {subreddit_name} ({sort}).")

                for post in posts:
                    if post.id not in seen_post_ids:
                        # Check if stock tickers appear in title or body
                        if contains_stock_mention(post.title) or contains_stock_mention(post.selftext):
                            logging.info(f"Post {post.id} contains stock ticker.")
                            post.comments.replace_more(limit=None)
                            all_comments = post.comments.list()

                            # Filter comments mentioning stocks
                            filtered_comments = [comment for comment in all_comments if contains_stock_mention(comment.body)]
                            if len(filtered_comments) > 500:
                                filtered_comments = filtered_comments[:500]  # Limit to 500 comments
                            seen_post_ids.add(post.id)  # Add post ID to seen list
                            logging.info(f"Found {len(filtered_comments)} relevant comments for post {post.id}.")
                            # Store only posts with relevant comments
                            post_id = post.id  # Assign a unique post ID

                            # Store post separately
                            data.append({
                                "type": "post",
                                "post_id": post_id,  # Unique identifier
                                "subreddit": subreddit_name,
                                "title": post.title,
                                "author": post.author.name if post.author else "[deleted]",
                                "url": post.url,
                                "score": post.score,
                                "body": post.selftext
                            })

                            # Store each filtered comment separately using the original comment object
                            for comment in filtered_comments:
                                if comment.id not in seen_post_ids:
                                    seen_post_ids.add(comment.id)  # Add comment ID to seen list
                                    data.append({
                                        "type": "comment",
                                        "post_id": comment.id,  # Link to the post
                                        "subreddit": subreddit_name,
                                        "title": post.title,
                                        "author": comment.author.name if comment.author else "[deleted]",
                                        "score": comment.score,
                                        "body": comment.body,
                                    })

                            total_count += len(filtered_comments) + 1  # +1 for the post itself
                    
                    if total_count >= target/len(subreddits):
                        break
                    # Stop if target posts are reached
                    elif len(data) >= target:
                        break
                    logging.info(f"Scraped {len(data)} posts so far...")

                if total_count >= target/len(subreddits):
                    logging.info(f"Finished scraping {subreddit_name} ({sort}). Total posts: {len(data)}")
                    break  # Exit the retry loop if successful
                else:
                    logging.info(f"Finished scraping {subreddit_name} ({sort}). Total posts: {len(data)}")
                    break  # Exit the retry loop if successful
                
            except prawcore.exceptions.TooManyRequests as e:
                wait_time = 2 ** (retries + 1)  # Exponential backoff
                logging.warning(f"Rate limit hit. Retrying {sort} in {wait_time} seconds... ({retries + 1}/{max_retries})")
                time.sleep(wait_time)
                retries += 1

    if len(data) >= target:
        break

# Save data to CSV
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"reddit_scraper_{timestamp}.csv"
# new_df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
# new_df.to_csv(filename, index=False)
df = pd.DataFrame(data)
df.to_csv(filename, index=False)

logging.info(f"Scraping completed. Data saved to {filename}. Total unique posts: {len(seen_post_ids)}")
logging.info("Reddit Scraper finished successfully.")
