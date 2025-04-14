# Otimizador de Agendamento de Pistas de Aeroporto

Este aplicativo Streamlit ajuda a otimizar o agendamento de pousos e decolagens em um aeroporto com múltiplas pistas. O objetivo é minimizar as penalidades por atraso, respeitando as restrições de segurança entre voos consecutivos.

## Características

- **Algoritmos construtivos**: Implementa três algoritmos gulosos diferentes (Earliest Ready Time, Least Penalty, Combined Heuristic)
- **Variable Neighborhood Descent (VND)**: Melhora as soluções iniciais com busca local
- **Visualização interativa**: Visualiza o agendamento das pistas e análise comparativa
- **Análise estatística**: Calcula métricas como valor da solução, tempos de execução e GAP para solução ótima
- **Exportação de solução**: Permite exportar a melhor solução encontrada

## Algoritmos Implementados

### Algoritmos Construtivos
- **Earliest Ready Time**: Agenda voos com base no tempo de liberação mais cedo
- **Least Penalty**: Minimiza penalidades por atraso
- **Combined Heuristic**: Combina considerações de tempo de liberação e penalidade

### Algoritmo de Busca Local
- **VND (Variable Neighborhood Descent)**: Explora diferentes estruturas de vizinhança para melhorar a solução inicial

### Estruturas de Vizinhança
- **Swap Flights**: Troca dois voos consecutivos na mesma pista
- **Move Flight**: Move um voo para uma posição diferente na mesma pista
- **Swap Between Runways**: Troca voos entre pistas diferentes

## Formato de Entrada

O aplicativo espera um arquivo de entrada com o seguinte formato:
```
number_of_flights
number_of_runways

array r (release times)
array c (processing times)
array p (penalties)

matrix t (waiting times)
```

## Instalação

```bash
pip install -r project_requirements.txt
```

## Execução

```bash
streamlit run app.py
```

## Exemplo

Um exemplo de arquivo de entrada:
```
6
2

5 25 15 40 75 50
15 25 20 30 15 25
55 90 61 120 45 50

0 10 15 8 21 15
10 0 10 13 15 20
17 9 0 10 14 8
11 13 12 0 10 10
5 10 15 20 0 12
5 10 15 20 28 0
```

## Autores

Desenvolvido para a disciplina de Análise e Projeto de Algoritmos da Universidade Federal da Paraíba.