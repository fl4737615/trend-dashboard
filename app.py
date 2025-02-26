# app.py
import os
import dash
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Configure NLTK data path for Render
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)
sia = SentimentIntensityAnalyzer()

# Initialize Dash app
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.MATERIA],
          assets_folder='assets/css',
          assets_url_path='css')
server = app.server

# --- Data Fetching Functions with Enhanced Error Handling ---
def fetch_wikipedia_views(topic="Climate_policy", days=30):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{topic}/daily/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data.get("items", []))
        if not df.empty:
            df["date"] = pd.to_datetime(df["timestamp"].str[:8])
            return df[["date", "views"]].set_index("date")
        return pd.DataFrame()
    except Exception as e:
        print(f"Wikipedia API Error: {e}")
        return pd.DataFrame()

def fetch_gdelt_news(query="climate change"):
    try:
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query} sourcecountry:US&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        
        df = pd.DataFrame([{
            "date": pd.to_datetime(a.get("seendate", datetime.now())).date(),
            "title": a.get("title", ""),
            "url": a.get("url", "")
        } for a in articles])
        
        if not df.empty:
            df["sentiment"] = df["title"].apply(lambda x: sia.polarity_scores(x)["compound"])
        return df.set_index("date")
    except Exception as e:
        print(f"GDELT API Error: {e}")
        return pd.DataFrame()

def fetch_stock_trend(ticker="ICLN"):
    try:
        df = yf.download(ticker, period="30d", progress=False)
        return df[["Close"]].rename(columns={"Close": "price"})
    except Exception as e:
        print(f"YFinance Error: {e}")
        return pd.DataFrame()

def fetch_arxiv_papers(query="carbon capture"):
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=50"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        entries = soup.find_all("entry")
        dates = [pd.to_datetime(entry.published.text).date() for entry in entries]
        return pd.DataFrame({"count": 1}, index=dates).resample("D").sum()
    except Exception as e:
        print(f"arXiv API Error: {e}")
        return pd.DataFrame()

# --- Data Processing ---
def get_merged_data():
    try:
        web = fetch_wikipedia_views()
        news = fetch_gdelt_news()
        finance = fetch_stock_trend()
        academic = fetch_arxiv_papers()
        
        merged = pd.concat([web, news, finance, academic], axis=1)
        merged.columns = ["Wikipedia_Views", "News_Sentiment", "Stock_Price", "Research_Papers"]
        merged.ffill(inplace=True)
        return merged.last("30D")
    except Exception as e:
        print(f"Data Merge Error: {e}")
        return pd.DataFrame()

# --- Responsive UI Layout ---
app.layout = dbc.Container(fluid=True, children=[
    dbc.NavbarSimple(
        brand="Trend Intelligence Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4("Wikipedia Traffic", className="card-title"),
            dcc.Graph(id='web-views', config={'displayModeBar': False})
        ], className="shadow-sm"), md=12, lg=6, className="mb-4"),
        
        dbc.Col(dbc.Card([
            html.H4("News Sentiment", className="card-title"),
            dcc.Graph(id='news-sentiment', config={'displayModeBar': False})
        ], className="shadow-sm"), md=12, lg=6, className="mb-4"),
        
        dbc.Col(dbc.Card([
            html.H4("Stock Trends", className="card-title"),
            dcc.Graph(id='stock-trend', config={'displayModeBar': False})
        ], className="shadow-sm"), md=12, lg=6, className="mb-4"),
        
        dbc.Col(dbc.Card([
            html.H4("Research Activity", className="card-title"),
            dcc.Graph(id='research-activity', config={'displayModeBar': False})
        ], className="shadow-sm"), md=12, lg=6, className="mb-4")
    ]),
    
    dcc.Interval(id='refresh', interval=3600*1000)
])

# --- Callbacks for Live Updates ---
@app.callback(
    [Output('web-views', 'figure'),
     Output('news-sentiment', 'figure'),
     Output('stock-trend', 'figure'),
     Output('research-activity', 'figure')],
    [Input('refresh', 'n_intervals')]
)
def update_all_charts(n):
    merged = get_merged_data()
    if merged.empty:
        return [px.scatter()] * 4
    
    figures = []
    for col in merged.columns:
        fig = px.line(
            merged,
            y=col,
            labels={'value': col.replace('_', ' '), 'index': 'Date'},
            title=f"{col.replace('_', ' ')} Trend"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=20)
        )
        figures.append(fig)
    
    return figures

if __name__ == "__main__":
    app.run_server(debug=False)
