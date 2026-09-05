from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import plotly.graph_objects as go

def am_exemplo_01(request):
    return HttpResponse("Exemplos de Aprendizado de Máquina (AM)")

def regressao_01(request):
    # Valor de partida (preço base do imóvel)
    a = 0
    
    # Quanto o preço aumenta para cada m² adicional
    b = 0.8
    
    # Área do imóvel (m²)
    X = np.linspace(0, 1000, 1000)
    
    # Preço estimado do imóvel (em milhares de reais)
    Y = a + b * X

    # Cria o gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X,
        y=Y,
        mode='lines',
        line=dict(color='red', width=3),
        name='Preço estimado'
    ))

    # Configuração do gráfico
    fig.update_layout(
        title='Preço estimado do imóvel',
        xaxis_title='Área (m²)',
        yaxis_title='Preço (1000 R$)',
        xaxis=dict(range=[0, 1000], tickmode='linear', dtick=100),
        yaxis=dict(range=[0, 2000]),
        template='plotly_white'
    )

    data = {}
    data['grafico'] = fig.to_html(full_html=False)
    return render(request, 'regressao_01.html', data)