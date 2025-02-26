import os
import nltk
import heapq
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ========== NLTK CONFIGURATION ========== #
NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=NLTK_DIR)

# ========== TREND DETECTION ENGINE ========== #
class TrendFinder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        
    def get_trends(self):
        with ThreadPoolExecutor() as executor:
            wiki = executor.submit(self._get_wiki_trends)
            news = executor.submit(self._get_news_entities)
            stocks = executor.submit(self._get_stock_movers)
            research = executor.submit(self._get_research_pulse)
            
            entities = wiki.result() + news.result() + stocks.result() + research.result()
        
        embeddings = self.model.encode(list(set(entities)))
        clusters = DBSCAN(eps=0.35, min_samples=2).fit_predict(embeddings)
        
        trend_scores = defaultdict(float)
        for idx, cluster in enumerate(clusters):
            if cluster != -1:
                entity = entities[idx]
                trend_scores[entity] += 1  # Base score
                trend_scores[entity] += self.sia.polarity_scores(entity)['compound']  # Sentiment boost
        
        return heapq.nlargest(25, trend_scores.items(), key=lambda x: x[1])

    def _get_wiki_trends(self):
        url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
        date = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
        try:
            return [item['article'] for item in requests.get(f"{url}{date}").json()['items'][0]['articles'][:25]]
        except:
            return []

    def _get_news_entities(self):
        try:
            data = requests.get("https://api.gdeltproject.org/api/v2/doc/doc?query=sourcecountry:US&format=json").json()
            return [art['title'] for art in data.get('articles', [])[:50]]
        except:
            return []

    def _get_stock_movers(self):
        try:
            return [f"{ticker} Stock" for ticker in yf.get_top_movers()['gainers'][:10] + yf.get_top_movers()['losers'][:10]]
        except:
            return []

    def _get_research_pulse(self):
        try:
            soup = BeautifulSoup(requests.get("http://export.arxiv.org/api/query?search_query=all&max_results=50").content, 'xml')
            return [entry.title.text for entry in soup.find_all('entry')]
        except:
            return []

# ========== RESPONSIVE UI ========== #
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dcc.Interval(id='refresh', interval=300*1000),  # 5 minutes
    dbc.Row(dbc.Col(html.H1("Global Trend Matrix", className="header"))),
    dbc.Row(id='trend-cards', className="g-2")
], fluid=True)

@app.callback(
    Output('trend-cards', 'children'),
    Input('refresh', 'n_intervals')
)
def update_cards(_):
    engine = TrendFinder()
    trends = engine.get_trends()
    
    return [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Div(trend, className="trend-title"),
                        html.Div(
                            className="confidence-bar",
                            style={'width': f'{min(score*25, 100)}%'}
                        )
                    ], className="header-wrapper")
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span(src, className="source-badge") 
                        for src in ['🌐', '📰', '💹', '📚'][:int(score)]
                    ], className="source-container")
                ])
            ], className="trend-card"),
            xs=12, sm=6, md=4, lg=3
        ) for trend, score in trends
    ]

if __name__ == "__main__":
    app.run_server(debug=False)
