import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Definir o caminho da pasta em que o projeto se encontra
folder_class_path = os.path.abspath(os.getcwd())

'''
Programa com o intuito de treinar o uso de diferentes tipos de gráficos - Sankey
Let's Code.

Dados utilizados:
- CSV Kaggle Survey 2021
'''

# https://www.kaggle.com/c/kaggle-survey-2021/data
def main():
    kaggle_df, descricao_colunas = receber_bases()
    df_analise = tratar_dados(kaggle_df)
    sankey_plot(df_analise, 
            columns = ['Genero', 'Pais', 'Linguagem', 'Cargo', 'Experiencia'], 
            color_at = ['Python', 'Man', 'Brazil', 'Under 1 year', '20 or more years'], 
            color = ['orange', 'blue', 'green', 'lightblue', 'red'])


# Função com o objetivo de receber as bases de dados csv
def receber_bases():
    kaggle_df = pd.read_csv(folder_class_path + '/kaggle_survey_2021_responses.csv', encoding='utf-8-sig', low_memory=False)
    desc = kaggle_df.iloc[0]
    kaggle_df = kaggle_df.iloc[1:]

    return kaggle_df, desc

def tratar_dados(kaggle_df):
    df_analise = kaggle_df[['Q2', 'Q3', 'Q5', 'Q8', 'Q15']].rename(columns = { 
        'Q2': 'Genero',
        'Q3': 'Pais',
        'Q5': 'Cargo',
        'Q8': 'Linguagem',
        'Q15': 'Experiencia'
    })
    other_countries = np.append(df_analise.groupby('Pais').size().sort_values().iloc[:-9].index.values, 'Other')
    other_languages = df_analise.groupby('Linguagem').size().sort_values().iloc[:-4].index.values
    other_jobs = np.append(df_analise.groupby('Cargo').size().sort_values().iloc[:-5].index.values, 'Other')
    df_analise = df_analise.replace({'Pais':{k:'Other Countries' for k in other_countries},
                    'Linguagem': {k:'Other Languages' for k in other_languages},
                    'Cargo': {k:'Other Jobs' for k in other_jobs},
                    'Experiencia': {'I do not use machine learning methods': 'No Experience'}})

    return df_analise


def get_data(df, input, output):
    data = (df
            .rename(columns = {input:'input', output:'output'})
            .groupby(['input', 'output'])
            .size()
            .to_frame('size')
            .reset_index()
    )
    return data


def sankey_plot(df, columns, color_at, color):
    labels = np.concatenate(df.apply(lambda x: x.unique(), axis = 0).values)
    labels = np.delete(labels, [24,30])
    labels = {v:k for k,v in enumerate(labels)}
    _df = df.replace(labels)
    data = []

    for i in range(len(columns) - 1):
        data.append(get_data(_df, columns[i], columns[i + 1]))
    data = pd.concat(data)
    data['color'] = 'lightgray'

    for c_at, c in zip(color_at, color):
        data.loc[(data[['input', 'output']] == labels[c_at]).any(axis = 1), 'color'] = c

    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = np.array(list(labels.keys()))[list(labels.values())],
            color = "lightgray"
            ),
        link = dict(
            source = data['input'], # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = data['output'],
            value = data['size'],
            color = data['color'])
    )])

    fig.update_layout()
    fig.show()


if __name__ == "__main__":
    main()