import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, Plus, X, Download, Pause, Play } from 'lucide-react';
import axios from 'axios';
import { StockData, RedditComment, FlattenedData, TimeRange } from './types';
import { flattenStockData, downloadCSV } from './utils/csvHandler';
import { DataTable } from './components/DataTable';

function App() {
  const [symbol, setSymbol] = useState('');
  const [subreddit, setSubreddit] = useState('');
  const [subreddits, setSubreddits] = useState<string[]>(['wallstreetbets']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [data, setData] = useState<StockData[]>([]);
  const [allScrapedData, setAllScrapedData] = useState<FlattenedData[]>([]);
  const [timeRange, setTimeRange] = useState<TimeRange>('day');
  const [isAutoScraping, setIsAutoScraping] = useState(false);
  const [scrapingStats, setScrapingStats] = useState({
    totalRequests: 0,
    totalPosts: 0,
    currentSubreddit: '',
    currentPage: 0
  });

  const addSubreddit = (e: React.FormEvent) => {
    e.preventDefault();
    if (subreddit && !subreddits.includes(subreddit.toLowerCase())) {
      setSubreddits([...subreddits, subreddit.toLowerCase()]);
      setSubreddit('');
    }
  };

  const removeSubreddit = (subredditToRemove: string) => {
    setSubreddits(subreddits.filter(s => s !== subredditToRemove));
  };

  const fetchRedditPosts = async (subreddit: string, symbol: string, after?: string) => {
    const params = new URLSearchParams({
      q: symbol,
      sort: 'new',
      limit: '100',
      restrict_sr: 'on',
      t: timeRange
    });
    if (after) params.append('after', after);

    const response = await axios.get(
      `https://www.reddit.com/r/${subreddit}/search.json?${params.toString()}`
    );

    setScrapingStats(prev => ({
      ...prev,
      totalRequests: prev.totalRequests + 1
    }));

    return response.data;
  };

  const scrapeRedditComments = async (
    symbol: string,
    startIndex: number = 0,
    lastAfter: { [key: string]: string } = {}
  ) => {
    if (!isAutoScraping) return;

    try {
      setLoading(true);
      setError('');

      let allComments: RedditComment[] = [];
      const maxPages = 25; // Increased max pages per subreddit

      for (let i = startIndex; i < subreddits.length; i++) {
        const subreddit = subreddits[i];
        let after = lastAfter[subreddit];
        let currentPage = 0;

        setScrapingStats(prev => ({
          ...prev,
          currentSubreddit: subreddit,
          currentPage: currentPage + 1
        }));

        while (isAutoScraping && currentPage < maxPages) {
          try {
            const data = await fetchRedditPosts(subreddit, symbol, after);
            const posts = data.data.children;
            
            if (posts.length === 0) break;

            const comments = posts
              .filter((post: any) => post.data.title)
              .map((post: any) => ({
                id: post.data.id,
                subreddit: post.data.subreddit,
                title: post.data.title,
                body: post.data.selftext || '',
                score: post.data.score,
                created_utc: post.data.created_utc,
                url: `https://reddit.com${post.data.permalink}`
              }));

            allComments = [...allComments, ...comments];
            after = data.data.after;
            currentPage++;

            setScrapingStats(prev => ({
              ...prev,
              totalPosts: prev.totalPosts + comments.length,
              currentPage: currentPage
            }));

            // Save the current batch of data
            const newData: StockData = {
              symbol: symbol.toUpperCase(),
              timestamp: new Date().toISOString(),
              subreddits: [subreddit],
              comments
            };

            const flattenedData = flattenStockData(newData);
            setAllScrapedData(prevData => [...prevData, ...flattenedData]);
            setData(prevData => [...prevData, newData]);

            if (!after) break;

            // Add delay to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 2000));
          } catch (error) {
            console.error(`Error fetching page ${currentPage} for ${subreddit}:`, error);
            await new Promise(resolve => setTimeout(resolve, 5000)); // Longer delay on error
            break;
          }
        }

        // If auto-scraping was stopped, save the current position
        if (!isAutoScraping) {
          return { lastSubredditIndex: i, lastAfter: { ...lastAfter, [subreddit]: after } };
        }
      }

      // If we've completed all subreddits, start over
      if (isAutoScraping) {
        await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds before starting over
        scrapeRedditComments(symbol, 0, {});
      }
    } catch (err) {
      setError('Error fetching data from Reddit');
      console.error(err);
      if (isAutoScraping) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        scrapeRedditComments(symbol, startIndex, lastAfter);
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleAutoScraping = () => {
    setIsAutoScraping(!isAutoScraping);
  };

  useEffect(() => {
    if (isAutoScraping && symbol.trim()) {
      scrapeRedditComments(symbol.trim());
    }
  }, [isAutoScraping]);

  const handleDownload = () => {
    if (allScrapedData.length > 0) {
      downloadCSV(allScrapedData);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center mb-8">
          <TrendingUp className="h-8 w-8 text-blue-600 mr-2" />
          <h1 className="text-3xl font-bold text-gray-900">Reddit Stock Scraper</h1>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="mb-6">
            <form onSubmit={addSubreddit} className="flex gap-4 mb-4">
              <div className="flex-1">
                <label htmlFor="subreddit" className="block text-sm font-medium text-gray-700 mb-2">
                  Add Subreddit
                </label>
                <input
                  type="text"
                  id="subreddit"
                  value={subreddit}
                  onChange={(e) => setSubreddit(e.target.value.toLowerCase())}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  placeholder="Enter subreddit name"
                />
              </div>
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 self-end"
              >
                <Plus className="h-5 w-5" />
              </button>
            </form>
            <div className="flex flex-wrap gap-2">
              {subreddits.map(sub => (
                <span
                  key={sub}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                >
                  r/{sub}
                  <button
                    onClick={() => removeSubreddit(sub)}
                    className="ml-2 text-blue-600 hover:text-blue-800"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </span>
              ))}
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Time Range
            </label>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as TimeRange)}
              className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="hour">Past Hour</option>
              <option value="day">Past 24 Hours</option>
              <option value="week">Past Week</option>
              <option value="month">Past Month</option>
              <option value="year">Past Year</option>
              <option value="all">All Time</option>
            </select>
          </div>

          <div className="flex gap-4">
            <div className="flex-1">
              <label htmlFor="symbol" className="block text-sm font-medium text-gray-700 mb-2">
                Stock Symbol
              </label>
              <div className="relative">
                <input
                  type="text"
                  id="symbol"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 pl-10"
                  placeholder="Enter stock symbol (e.g., NVDA)"
                />
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
              </div>
            </div>
            <button
              type="button"
              onClick={toggleAutoScraping}
              disabled={loading || subreddits.length === 0 || !symbol.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 self-end"
            >
              {loading ? (
                'Scraping...'
              ) : isAutoScraping ? (
                <><Pause className="h-5 w-5 mr-2" />Pause Scraping</>
              ) : (
                <><Play className="h-5 w-5 mr-2" />Start Scraping</>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-blue-800">Total API Requests</h3>
              <p className="text-3xl font-bold text-blue-600">{scrapingStats.totalRequests}</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-green-800">Total Posts Collected</h3>
              <p className="text-3xl font-bold text-green-600">{scrapingStats.totalPosts}</p>
            </div>
          </div>
          {loading && (
            <div className="text-sm text-gray-600">
              Currently scraping: r/{scrapingStats.currentSubreddit} (Page {scrapingStats.currentPage})
            </div>
          )}
        </div>

        {data.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Search History</h2>
              <button
                onClick={handleDownload}
                className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                <Download className="h-5 w-5 mr-2" />
                Download CSV ({allScrapedData.length} posts)
              </button>
            </div>
            <DataTable data={data} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;