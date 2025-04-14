import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, TrendingUp, TrendingDown, LineChart, MessageSquare, Hash, Database, Tag, AlertCircle } from 'lucide-react';

interface Comment {
  text: string;
  platform: string;
}

interface Keyword {
  text: string;
  count: number;
}

function App() {
  const [query, setQuery] = useState('');
  const [queryError, setQueryError] = useState('');
  const [sentiment, setSentiment] = useState<'bullish' | 'bearish' | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [comments, setComments] = useState<Comment[]>([]);
  const [subreddits, setSubreddits] = useState<string[]>([]);
  const [recordCount, setRecordCount] = useState<number>(0);
  const [keywords, setKeywords] = useState<Keyword[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    //Query validation here
    if(!query.trim()) {
      setQueryError('Please enter a valid query');
      return;
    }
    setQueryError('');

    setIsLoading(true);
    
    // THIS PORTION ADDS IN DUMMY DATA. 
    setTimeout(() => {
      setSentiment(Math.random() > 0.5 ? 'bullish' : 'bearish');
      setComments([
        { text: "IBKR's interface is much better for trading options", platform: "IBKR" },
        { text: "Tiger's mobile app has improved significantly", platform: "TigerBrokers" },
        { text: "The execution speed on IBKR is unmatched", platform: "IBKR" },
      ]);
      setSubreddits(['r/investing', 'r/stocks', 'r/options']);
      setRecordCount(1234);
      setKeywords([
        { text: "interface", count: 45 },
        { text: "trading", count: 38 },
        { text: "mobile", count: 32 },
        { text: "execution", count: 28 },
        { text: "speed", count: 25 },
      ]);
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <div className="container mx-auto px-4 py-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-12"
        >
          <div className="flex items-center gap-2">
            <LineChart className="w-8 h-8 text-blue-400" />
            <h1 className="text-2xl font-bold">Trading Platform Sentiment Analyzer</h1>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gray-800 rounded-xl p-8 shadow-2xl mb-8"
        >
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter Query"
                className="w-full pl-12 pr-4 py-3 bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              />
              <motion.button
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-500 text-white px-4 py-1 rounded-md hover:bg-blue-600 transition-colors"
                disabled={isLoading}
              >
                Analyze
              </motion.button>
            </div>
            {queryError && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 mt-2 text-red-400"
              >
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{queryError}</span>
              </motion.div>
            )}
          </form>

          {isLoading ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-center items-center py-12"
            >
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </motion.div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Sentiment Section */}
              {sentiment && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center"
                >
                  <motion.div
                    className={`inline-flex items-center justify-center p-8 rounded-full ${
                      sentiment === 'bullish' ? 'bg-green-500/20' : 'bg-red-500/20'
                    }`}
                  >
                    {sentiment === 'bullish' ? (
                      <TrendingUp className="w-24 h-24 text-green-500" />
                    ) : (
                      <TrendingDown className="w-24 h-24 text-red-500" />
                    )}
                  </motion.div>
                  <motion.h2
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`text-3xl font-bold mt-4 ${
                      sentiment === 'bullish' ? 'text-green-500' : 'text-red-500'
                    }`}
                  >
                    {sentiment.toUpperCase()}
                  </motion.h2>
                  <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-gray-400 mt-2"
                  >
                    Platform sentiment analysis for {query}
                  </motion.p>
                </motion.div>
              )}

              {/* Record Count Widget */}
              {recordCount > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-700/50 rounded-xl p-6"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <Database className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-xl font-semibold">Relevant Records</h3>
                  </div>
                  <div className="text-center">
                    <motion.span
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className="text-5xl font-bold text-emerald-400"
                    >
                      {recordCount.toLocaleString()}
                    </motion.span>
                    <p className="text-gray-400 mt-2">Total records analyzed</p>
                  </div>
                </motion.div>
              )}

              {/* Keywords Widget */}
              {keywords.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-700/50 rounded-xl p-6"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <Tag className="w-5 h-5 text-yellow-400" />
                    <h3 className="text-xl font-semibold">Top Keywords</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {keywords.map((keyword, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-800/50 px-3 py-1 rounded-full"
                      >
                        <span className="text-gray-300">{keyword.text}</span>
                        <span className="ml-2 text-yellow-400">{keyword.count}</span>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Comments Section */}
              {comments.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-700/50 rounded-xl p-6"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <MessageSquare className="w-5 h-5 text-blue-400" />
                    <h3 className="text-xl font-semibold">Trading Platform Comments</h3>
                    <span className="ml-auto bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full text-sm">
                      {comments.length} comments
                    </span>
                  </div>
                  <div className="space-y-4 max-h-80 overflow-y-auto custom-scrollbar">
                    {comments.map((comment, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-800/50 rounded-lg p-4"
                      >
                        <p className="text-gray-300">{comment.text}</p>
                        <span className="text-sm text-blue-400 mt-2 inline-block">
                          {comment.platform}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Subreddits Section */}
              {subreddits.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-gray-700/50 rounded-xl p-6"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <Hash className="w-5 h-5 text-purple-400" />
                    <h3 className="text-xl font-semibold">Related Subreddits</h3>
                    <span className="ml-auto bg-purple-500/20 text-purple-400 px-2 py-1 rounded-full text-sm">
                      {subreddits.length} subreddits
                    </span>
                  </div>
                  <div className="space-y-4 max-h-80 overflow-y-auto custom-scrollbar">
                    {subreddits.map((subreddit, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-gray-800/50 rounded-lg p-4"
                      >
                        <p className="text-gray-300">{subreddit}</p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}

export default App;