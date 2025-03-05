import Papa from 'papaparse';
import { StockData, FlattenedData } from '../types';

export const flattenStockData = (data: StockData): FlattenedData[] => {
  return data.comments.map(comment => ({
    symbol: data.symbol,
    timestamp: data.timestamp,
    subreddit: comment.subreddit,
    post_id: comment.id,
    title: comment.title,
    body: comment.body,
    score: comment.score,
    created_utc: new Date(comment.created_utc * 1000).toISOString(),
    url: comment.url
  }));
};

export const downloadCSV = (data: FlattenedData[]) => {
  try {
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `reddit_stock_data_${new Date().toISOString()}.csv`;
    link.click();
  } catch (error) {
    console.error('Error downloading CSV:', error);
    throw error;
  }
};