import os
import nltk
import heapq
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)

# ========== NLTK CONFIGURATION ========== #
NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=NLTK_DIR)

# ========== ROBUST TREND ENGINE ========== #
class TrendRadar:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def get_trends(self):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._get_wiki_trends): "Wikipedia",
                executor.submit(self._get_news_entities): "News",
                executor.submit(self._get_stock_signals): "Stocks",
                executor.submit(self._get_research_pulse): "Research"
            }
            
            entities = []
            for future in futures:
                try:
                    entities += future.result()
                except Exception as e:
                    logging.error(f"{futures[future]} scan failed: {str(e)}")
        
        return self._analyze_trends(entities)
    
    def _analyze_trends(self, entities):
        try:
            embeddings = self.model.encode(list(set(entities)))
            clusters = DBSCAN(eps=0.35, min_samples=2).fit_predict(embeddings)
            
            trend_scores = defaultdict(float)
            for idx, cluster in enumerate(clusters):
                if cluster != -1:
                    entity = entities[idx]
                    trend_scores[entity] += 1
                    trend_scores[entity] += self.sia.polarity_scores(entity)['compound']
            
            return heapq.nlargest(25, trend_scores.items(), key=lambda x: x[1])
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return []

    # Data collection methods
    def _get_wiki_trends(self):
        try:
            date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date}"
            return [item['article'] for item in requests.get(url, timeout=10).json()['items'][0]['articles'][:25]]
        except Exception as e:
            logging.warning(f"Wikipedia failed: {str(e)}")
            return []

    def _get_news_entities(self):
        entities = []
        # GDELT (always available)
        try:
            data = requests.get("https://api.gdeltproject.org/api/v2/doc/doc?query=sourcecountry:US&format=json", timeout=15).json()
            entities += [art['title'] for art in data.get('articles', [])[:30]]
        except Exception as e:
            logging.warning(f"GDELT failed: {str(e)}")
        
        # NewsAPI (optional)
        newsapi_key = os.getenv('NEWSAPI_KEY', '')
        if newsapi_key and newsapi_key != 'your_newsapi_key':
            try:
                url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi_key}"
                newsapi_data = requests.get(url, timeout=10).json()
                entities += [art['title'] for art in newsapi_data.get('articles', [])[:20]]
            except Exception as e:
                logging.warning(f"NewsAPI failed: {str(e)}")
        
        return entities

    def _get_stock_signals(self):
        entities = []
        av_key = os.getenv('ALPHAVANTAGE_KEY', '')
        
        # AlphaVantage (preferred)
        if av_key and av_key != 'your_alphavantage_key':
            try:
                url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={av_key}"
                data = requests.get(url, timeout=10).json()
                gainers = data.get('top_gainers', [])[:3]
                losers = data.get('top_losers', [])[:3]
                entities += [entry['ticker'] for entry in gainers + losers]
            except Exception as e:
                logging.warning(f"AlphaVantage failed: {str(e)}")
        
        # Yahoo Finance fallback
        try:
            tickers = ['TSLA', 'AAPL', 'AMZN', 'GOOG', 'META']  # Default watchlist
            data = yf.download(tickers, period='1d', group_by='ticker')
            entities += [f"{ticker} Stock" for ticker in tickers]
        except Exception as e:
            logging.warning(f"Yahoo Finance failed: {str(e)}")
        
        return entities

    def _get_research_pulse(self):
        try:
            soup = BeautifulSoup(requests.get("http://export.arxiv.org/api/query?search_query=all&max_results=50", timeout=15).content, 'xml')
            return [entry.title.text for entry in soup.find_all('entry')[:30]]
        except Exception as e:
            logging.warning(f"arXiv failed: {str(e)}")
            return []

# ========== SELF-HEALING UI ========== #
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dcc.Interval(id='refresh', interval=300*1000),  # 5 minutes
    dbc.Row(dbc.Col(html.H1("Global Trend Radar", className="header"))),
    dbc.Row(id='trend-grid', className="g-2")
], fluid=True)

@app.callback(
    Output('trend-grid', 'children'),
    Input('refresh', 'n_intervals')
)
def update_interface(n):
    try:
        engine = TrendRadar()
        trends = engine.get_trends()
        return create_cards(trends)
    except Exception as e:
        logging.error(f"Critical failure: {str(e)}")
        return error_card()

def create_cards(trends):
    return [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Div(trend, className="trend-title"),
                        html.Div(className="confidence-bar", style={'width': f'{min(score*25, 100)}%'})
                    ])
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span(src, className="source-badge") 
                        for src in ['🌐', '📰', '💹', '📚'][:int(score)]
                    ])
                ])
            ], className="trend-card"),
            xs=12, sm=6, md=4, lg=3
        ) for trend, score in trends
    ] if trends else error_card()

def error_card():
    return dbc.Col(
        dbc.Card([
            dbc.CardBody("⚠️ Temporary data outage - next scan in 5 minutes")
        ], className="error-card"),
        width=12
    )

if __name__ == "__main__":
    app.run_server(debug=False)
