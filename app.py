import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sys
sys.path.append(r'C:\Users\Naved\OneDrive\Desktop\Sentiment Analysis\Sentiment Analysis Project\Sentiment Analysis Project\financial_sentiment_analysis')
from src.sentiment_analyzer.hybrid_analyzer import HybridSentimentAnalyzer
from src.models.vader_model import VADERModel
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'analyzed_financial_news_data.csv')
   
    df = pd.read_csv(csv_path)
    df['publishedDate'] = pd.to_datetime(df['publishedDate'], format='ISO8601', errors='coerce')

    analyzer = HybridSentimentAnalyzer(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ml_model.joblib'))
    vader_model = VADERModel()

    app.layout = dbc.Container([                        #Dashboard layout
        dbc.Row([
            dbc.Col([
                html.H1("Financial Sentiment Analysis Dashboard", className="text-center mb-4"),
                html.Hr(),
            ], width=12)
        ]), 
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Select Company", className="card-title"),
                        dcc.Dropdown(
                            id='company-dropdown',
                            options=[{'label': company, 'value': company} for company in df['companyName'].unique()],
                            value=df['companyName'].unique()[0],
                            className="mb-3"
                        ),
                    ])
                ], className="mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Real-time Sentiment Analysis", className="card-title"),
                        dcc.Textarea(
                            id='text-input',
                            placeholder='Enter text for real-time sentiment analysis...',
                            style={'width': '100%', 'height': 100},
                            className="mb-3"
                        ),
                        dbc.Button("Analyze", id="analyze-button", color="primary", className="mb-3"),
                        html.Div(id='sentiment-output')
                    ])
                ]),
            ], md=4),
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Sentiment Trend", tab_id="trend"),
                    dbc.Tab(label="Sentiment Distribution", tab_id="distribution"),
                    dbc.Tab(label="Sentiment Comparison", tab_id="comparison"),
                ], id="tabs", active_tab="trend"),
                html.Div(id="tab-content", className="p-4")
            ], md=8),
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Recent News", className="mt-4"),
                html.Div(id='recent-news')
            ], width=12)
        ])
    ], fluid=True, style={'backgroundColor': '#90A4AE'})

    @app.callback(                                              #tab switching and visualization
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"),
         Input('company-dropdown', 'value')]
    )
    def render_tab_content(active_tab, selected_company):           #Plotting visual charts for trend, distribution and comparison
        if active_tab == "trend":
            return dcc.Graph(figure=update_sentiment_trend(selected_company))
        elif active_tab == "distribution":
            return dcc.Graph(figure=update_sentiment_distribution(selected_company))
        elif active_tab == "comparison":
            return dcc.Graph(figure=update_sentiment_comparison(selected_company))

    def update_sentiment_trend(selected_company):               #takes user text and analyzes sentiment using hybrid model
        filtered_df = df[df['companyName'] == selected_company].sort_values('publishedDate')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df['publishedDate'], y=filtered_df['sentiment_score'],
                                 mode='lines', name='Hybrid Score'))
        fig.add_trace(go.Scatter(x=filtered_df['publishedDate'], y=filtered_df['vader_score'],
                                 mode='lines', name='VADER Score'))
        fig.update_layout(title=f'Sentiment Trend for {selected_company}',
                          xaxis_title='Date',
                          yaxis_title='Sentiment Score',
                          legend_title='Model',
                          hovermode='x unified')
        return fig

    def update_sentiment_distribution(selected_company):
        filtered_df = df[df['companyName'] == selected_company]
        hybrid_counts = filtered_df['hybrid_sentiment'].value_counts()
        vader_counts = filtered_df['vader_sentiment'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(labels=['Positive', 'Neutral', 'Negative'], values=[hybrid_counts.get('Positive', 0), hybrid_counts.get('Neutral', 0), hybrid_counts.get('Negative', 0)], name='Hybrid', domain={'x': [0, 0.48]}),
            go.Pie(labels=['Positive', 'Neutral', 'Negative'], values=[vader_counts.get('Positive', 0), vader_counts.get('Neutral', 0), vader_counts.get('Negative', 0)], name='VADER', domain={'x': [0.52, 1]})
        ])
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        fig.update_layout(
            title_text=f"Sentiment Distribution for {selected_company}",
            annotations=[dict(text='Hybrid', x=0.18, y=0.5, font_size=20, showarrow=False),
                         dict(text='VADER', x=0.82, y=0.5, font_size=20, showarrow=False)]
        )
        return fig

    def update_sentiment_comparison(selected_company):
        filtered_df = df[df['companyName'] == selected_company]
        sentiment_counts = pd.DataFrame({
            'Sentiment Type': ['Rule-based', 'ML-based', 'Unsupervised', 'Hybrid', 'VADER'],
            'Positive': [
                (filtered_df['rule_based_sentiment'] == 'Positive').sum(),
                (filtered_df['ml_sentiment'] == 'Positive').sum(),
                (filtered_df['unsupervised_sentiment'] == 'Cluster 0').sum(),
                (filtered_df['hybrid_sentiment'] == 'Positive').sum(),
                (filtered_df['vader_sentiment'] == 'Positive').sum()
            ],
            'Neutral': [
                (filtered_df['rule_based_sentiment'] == 'Neutral').sum(),
                (filtered_df['ml_sentiment'] == 'Neutral').sum(),
                (filtered_df['unsupervised_sentiment'] == 'Cluster 1').sum(),
                (filtered_df['hybrid_sentiment'] == 'Neutral').sum(),
                (filtered_df['vader_sentiment'] == 'Neutral').sum()
            ],
            'Negative': [
                (filtered_df['rule_based_sentiment'] == 'Negative').sum(),
                (filtered_df['ml_sentiment'] == 'Negative').sum(),
                (filtered_df['unsupervised_sentiment'] == 'Cluster 2').sum(),
                (filtered_df['hybrid_sentiment'] == 'Negative').sum(),
                (filtered_df['vader_sentiment'] == 'Negative').sum()
            ]
        })
        
        fig = px.bar(sentiment_counts, x='Sentiment Type', y=['Positive', 'Neutral', 'Negative'],
                     title=f'Sentiment Comparison for {selected_company}',
                     barmode='group',
                     color_discrete_sequence=['#28a745', '#ffc107', '#dc3545'])
        fig.update_layout(legend_title_text='Sentiment')
        return fig

    @app.callback(
        Output('recent-news', 'children'),
        Input('company-dropdown', 'value')
    )
    def update_recent_news(selected_company):               #Displays latest news for selected company with sentiment labels
        filtered_df = df[df['companyName'] == selected_company].sort_values('publishedDate', ascending=False).head(5)
        news_items = []
        for _, row in filtered_df.iterrows():
            news_items.append(dbc.Card(
                dbc.CardBody([
                    html.H5(row['webTitle'], className="card-title"),
                    html.P(f"Date: {row['publishedDate']}", className="card-text"),
                    html.P([
                        html.Strong("Rule-based: "),
                        html.Span(row['rule_based_sentiment'], className=f"badge bg-{get_sentiment_color(row['rule_based_sentiment'])} me-1"),
                        html.Strong("ML: "),
                        html.Span(row['ml_sentiment'], className=f"badge bg-{get_sentiment_color(row['ml_sentiment'])} me-1"),
                        html.Strong("Unsupervised: "),
                        html.Span(row['unsupervised_sentiment'], className=f"badge bg-{get_sentiment_color(row['unsupervised_sentiment'])} me-1"),
                        html.Strong("Hybrid: "),
                        html.Span(row['hybrid_sentiment'], className=f"badge bg-{get_sentiment_color(row['hybrid_sentiment'])} me-1"),
                        html.Strong("VADER: "),
                        html.Span(row['vader_sentiment'], className=f"badge bg-{get_sentiment_color(row['vader_sentiment'])} me-1"),
                    ], className="card-text"),
                    dbc.Button("Read more", href=row['webUrl'], target="_blank", color="primary", size="sm")
                ])
            , className="mb-3"))
        return news_items

    @app.callback(
        Output('sentiment-output', 'children'),
        Input('analyze-button', 'n_clicks'),
        State('text-input', 'value')
    )
    def update_sentiment_output(n_clicks, text):
        if n_clicks is None or not text:
            return ""
        
        hybrid_sentiment = analyzer.analyze_sentiment(text)
        hybrid_score = analyzer.get_sentiment_score(text)
        vader_sentiment = vader_model.analyze_sentiment(text)
        vader_score = vader_model.get_sentiment_score(text)
        
        return [
            dbc.Alert([
                html.H5("Hybrid Sentiment Analysis"),
                html.P(f"Sentiment: {hybrid_sentiment}"),
                html.P(f"Score: {hybrid_score:.2f}")
            ], color=get_sentiment_color(hybrid_sentiment), className="mb-3"),
            dbc.Alert([
                html.H5("VADER Sentiment Analysis"),
                html.P(f"Sentiment: {vader_sentiment}"),
                html.P(f"Score: {vader_score:.2f}")
            ], color=get_sentiment_color(vader_sentiment), className="mb-3")
        ]

    def get_sentiment_color(sentiment):
        if sentiment == 'Positive' or sentiment == 'Cluster 0':
            return 'success'
        elif sentiment == 'Negative' or sentiment == 'Cluster 2':
            return 'danger'
        else:
            return 'warning'

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)