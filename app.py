import os
import nltk
import heapq
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict

# ========== NLTK CONFIG ========== #
NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=NLTK_DIR)

# ========== TREND ENGINE ========== #
class TrendFinder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    
    def get_top_trends(self):
        entities = self._wiki_top() + self._news_entities() + \
                 self._stock_signals() + self._research_pulse()
        
        embeddings = self.model.encode(list(set(entities)))
        clusters = DBSCAN(min_samples=2, eps=0.3).fit_predict(embeddings)
        
        trend_scores = defaultdict(float)
        for idx, cluster in enumerate(clusters):
            if cluster != -1:
                entity = entities[idx]
                trend_scores[entity] += 1
                trend_scores[entity] += self.sia.polarity_scores(entity)['compound']
        
        return heapq.nlargest(25, trend_scores.items(), key=lambda x: x[1])

# (Data collection methods remain same as previous)

# ========== OPTIMIZED UI ========== #
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

app.layout = dbc.Container([
    dcc.Interval(id='refresh', interval=300*1000),
    dbc.Row(dbc.Col(html.H1("Global Trend Matrix", className="header-text"))),
    dbc.Row(id='trend-grid', className="g-2")
], fluid=True, className="main-container")

@callback(
    Output('trend-grid', 'children'),
    Input('refresh', 'n_intervals')
)
def update_grid(n):
    engine = TrendFinder()
    trends = engine.get_top_trends()
    
    return [
        dbc.Col(
            dbc.Card([
                html.Div(
                    className="confidence-meter",
                    children=html.Div(
                        className="confidence-fill",
                        style={'width': f'{min(score*25, 100)}%'}
                    )
                ),
                dbc.CardBody([
                    html.H5(trend, className="trend-title"),
                    html.Div([
                        html.Span(src, className="badge") 
                        for src in ['📈', '📰', '💹', '📚'][:int(score)]
                    ], className="source-badges")
                ])
            ], className="trend-card"),
            xs=12, sm=6, md=4, lg=3
        ) for trend, score in trends
    ]

if __name__ == "__main__":
    app.run_server(debug=False)
