"""
Debug Wikipedia S&P 500 fetch
"""

import pandas as pd
import urllib.request

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Add user agent to avoid 403 Forbidden error
req = urllib.request.Request(url)
req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

# Read HTML with custom request
with urllib.request.urlopen(req) as response:
    tables = pd.read_html(response.read())

print(f"Found {len(tables)} tables")

for i, table in enumerate(tables):
    print(f"\n{'='*80}")
    print(f"TABLE {i}")
    print(f"{'='*80}")
    print(f"Shape: {table.shape}")
    print(f"Columns: {list(table.columns)}")
    print(f"First 5 rows:")
    print(table.head(5))
    
    # Check if this looks like the S&P 500 table
    if 'Symbol' in table.columns or 'Ticker' in table.columns:
        print(f"\n*** THIS LOOKS LIKE THE S&P 500 TABLE ***")
        if 'Symbol' in table.columns:
            print(f"First 10 symbols: {table['Symbol'].head(10).tolist()}")
