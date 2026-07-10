"""
Interactive charts for optimization visualization.

Uses Plotly to create responsive, interactive visualizations of
compression, accuracy, latency, and other trade-offs.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import streamlit as st
from simulator.optimizer import OptimizationHistory


def create_tradeoff_scatter(
    accuracies: List[float],
    sizes_mb: List[float],
    compressions: List[float],
    prunings: List[float],
    title: str = "Accuracy vs Model Size Trade-off"
) -> go.Figure:
    """
    Create scatter plot of accuracy vs model size.
    
    Args:
        accuracies: List of accuracies
        sizes_mb: List of model sizes
        compressions: List of compression values
        prunings: List of pruning values
        title: Chart title
    
    Returns:
        Plotly figure
    """
    # Color by compression level
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sizes_mb,
        y=accuracies,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=compressions,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Compression"),
            line=dict(width=1, color='white')
        ),
        line=dict(color='rgba(0,0,0,0.2)'),
        text=[f"Compression: {c:.0%}<br>Pruning: {p:.0%}" 
              for c, p in zip(compressions, prunings)],
        hovertemplate='<b>Trade-off Point</b><br>Size: %{x:.1f}MB<br>Accuracy: %{y:.1%}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model Size (MB)",
        yaxis_title="Accuracy",
        hovermode='closest',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_metric_evolution(
    compressions: List[float],
    accuracies: List[float],
    sizes_mb: List[float],
    latencies_ms: List[float]
) -> go.Figure:
    """
    Create plot showing how metrics evolve with compression.
    
    Args:
        compressions: List of compression levels
        accuracies: List of accuracies
        sizes_mb: List of model sizes
        latencies_ms: List of latencies
    
    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Accuracy", "Model Size", "Latency")
    )
    
    # Accuracy vs Compression
    fig.add_trace(
        go.Scatter(
            x=compressions,
            y=accuracies,
            name='Accuracy',
            line=dict(color='#636EFA', width=2),
            marker=dict(size=6),
            hovertemplate='Compression: %{x:.0%}<br>Accuracy: %{y:.1%}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Size vs Compression
    fig.add_trace(
        go.Scatter(
            x=compressions,
            y=sizes_mb,
            name='Model Size',
            line=dict(color='#EF553B', width=2),
            marker=dict(size=6),
            hovertemplate='Compression: %{x:.0%}<br>Size: %{y:.1f}MB<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Latency vs Compression
    fig.add_trace(
        go.Scatter(
            x=compressions,
            y=latencies_ms,
            name='Latency',
            line=dict(color='#00CC96', width=2),
            marker=dict(size=6),
            hovertemplate='Compression: %{x:.0%}<br>Latency: %{y:.1f}ms<extra></extra>'
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Compression Level", row=1, col=1)
    fig.update_xaxes(title_text="Compression Level", row=1, col=2)
    fig.update_xaxes(title_text="Compression Level", row=1, col=3)
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Size (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=3)
    
    fig.update_layout(height=400, template='plotly_white', hovermode='x unified')
    
    return fig


def create_deployment_gauge(score: float, max_score: float = 100) -> go.Figure:
    """
    Create gauge chart for deployment readiness score.
    
    Args:
        score: Current score (0-100)
        max_score: Maximum possible score
    
    Returns:
        Plotly gauge figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Deployment Readiness"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, max_score]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 60], 'color': "#ffffcc"},
                {'range': [60, 80], 'color': "#ccffcc"},
                {'range': [80, 100], 'color': "#ccffff"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=400, template='plotly_white')
    return fig


def create_constraint_radar(progress: dict) -> go.Figure:
    """
    Create radar chart for constraint progress.
    
    Args:
        progress: Progress dictionary from scorer
    
    Returns:
        Plotly radar figure
    """
    categories = ['Accuracy', 'Size', 'Latency']
    values = [
        min(100, progress['accuracy']['progress_pct']),
        progress['size']['progress_pct'],
        progress['latency']['progress_pct']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Progress',
        line=dict(color='#636EFA'),
        fillcolor='rgba(99, 110, 250, 0.3)'
    ))
    
    # Add 100% reference
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100],
        theta=categories,
        fill=None,
        name='Target',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_optimization_history_chart(history: OptimizationHistory) -> go.Figure:
    """
    Create chart showing optimization history.
    
    Args:
        history: Optimization history
    
    Returns:
        Plotly figure
    """
    from plotly.subplots import make_subplots
    
    if not history.accuracies:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text="No optimization history yet", showarrow=False)
        return fig
    
    steps = list(range(len(history.accuracies)))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy", "Model Size", "Latency", "Compression Level")
    )
    
    # Accuracy over time
    fig.add_trace(
        go.Scatter(y=history.accuracies, name='Accuracy', line=dict(color='#636EFA')),
        row=1, col=1
    )
    
    # Size over time
    fig.add_trace(
        go.Scatter(y=history.sizes_mb, name='Size', line=dict(color='#EF553B')),
        row=1, col=2
    )
    
    # Latency over time
    fig.add_trace(
        go.Scatter(y=history.latencies_ms, name='Latency', line=dict(color='#00CC96')),
        row=2, col=1
    )
    
    # Compression level over time
    fig.add_trace(
        go.Scatter(y=history.compressions, name='Compression', line=dict(color='#AB63FA')),
        row=2, col=2
    )
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Size (MB)", row=1, col=2)
    fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Compression", row=2, col=2)
    
    fig.update_layout(height=600, template='plotly_white', hovermode='x unified')
    
    return fig


def display_pareto_frontier(
    accuracies: List[float],
    sizes_mb: List[float]
) -> go.Figure:
    """
    Create Pareto frontier visualization showing optimal trade-offs.
    
    Args:
        accuracies: List of accuracies
        sizes_mb: List of model sizes
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # All points
    fig.add_trace(go.Scatter(
        x=sizes_mb,
        y=accuracies,
        mode='markers',
        marker=dict(size=8, color='lightblue', opacity=0.6),
        name='All optimizations',
        hovertemplate='Size: %{x:.1f}MB<br>Accuracy: %{y:.1%}<extra></extra>'
    ))
    
    # Find Pareto frontier (dominated solutions removed)
    frontier_indices = []
    for i in range(len(sizes_mb)):
        is_dominated = False
        for j in range(len(sizes_mb)):
            if i != j:
                # Point j dominates i if: smaller size AND higher accuracy
                if sizes_mb[j] <= sizes_mb[i] and accuracies[j] >= accuracies[i]:
                    if sizes_mb[j] < sizes_mb[i] or accuracies[j] > accuracies[i]:
                        is_dominated = True
                        break
        if not is_dominated:
            frontier_indices.append(i)
    
    if frontier_indices:
        frontier_sizes = [sizes_mb[i] for i in frontier_indices]
        frontier_accuracies = [accuracies[i] for i in frontier_indices]
        
        # Sort by size for proper line drawing
        sorted_pairs = sorted(zip(frontier_sizes, frontier_accuracies))
        frontier_sizes = [p[0] for p in sorted_pairs]
        frontier_accuracies = [p[1] for p in sorted_pairs]
        
        fig.add_trace(go.Scatter(
            x=frontier_sizes,
            y=frontier_accuracies,
            mode='markers+lines',
            marker=dict(size=10, color='darkblue'),
            line=dict(color='darkblue', width=2),
            name='Pareto frontier',
            hovertemplate='Size: %{x:.1f}MB<br>Accuracy: %{y:.1%}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Pareto Frontier: Optimal Accuracy vs Size Trade-offs",
        xaxis_title="Model Size (MB)",
        yaxis_title="Accuracy",
        height=400,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig
