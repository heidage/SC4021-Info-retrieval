import pandas as pd

class SP500DataLoader:
    def __init__(self):
        """Fetch stock tickers from the Wikipedia page for S&P 500 companies."""
        url = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            # Read HTML tables from Wikipedia
            tables = pd.read_html(url)
            sp500_table = tables[0]  # The first table usually contains the relevant data
            
            # Extract ticker symbols
            self.tickers = sp500_table["Symbol"].astype(str).str.strip().tolist()
        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            self.tickers = []  # Return an empty list on failure

    def get_ticker_list(self):
        """Return a copy of the S&P 500 stock ticker list."""
        return self.tickers.copy()

if __name__ == "__main__":
    loader = SP500DataLoader()
    print(loader.get_ticker_list())