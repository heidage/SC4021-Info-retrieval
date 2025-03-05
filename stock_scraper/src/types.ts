export interface RedditComment {
  id: string;
  subreddit: string;
  title: string;
  body: string;
  score: number;
  created_utc: number;
  url: string;
}

export interface StockData {
  symbol: string;
  timestamp: string;
  subreddits: string[];
  comments: RedditComment[];
}

export interface FlattenedData {
  symbol: string;
  timestamp: string;
  subreddit: string;
  post_id: string;
  title: string;
  body: string;
  score: number;
  created_utc: string;
  url: string;
}

export type TimeRange = 'hour' | 'day' | 'week' | 'month' | 'year' | 'all';