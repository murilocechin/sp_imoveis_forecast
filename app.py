"""
Criando aplicação visual com a biblioteca Dash.
"""

# Importando Blibliotecas
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd 
import joblib

# Carregando modelos finais treinados
modelo_aluguel_final = joblib.load("modelagem/rf_rent_final.pkl")
modelo_venda_final = joblib.load("modelagem/xgb_sale_final.pkl")


# Iniciando aplicativo
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Função para o layout da página "Início"
def layout_inicio():

    # Definindo cards de redes sociais
    # Linkedin 
    card1 = html.Div([html.A(
            "Meu Perfil no Linkedin",
            href="https://www.linkedin.com/in/murilo-cechin",  
            style={"color": "white", "text-decoration": "none","text-shadow": "1px 1px 2px black"},
            target="_blank"  
        )], style={
        "height": "60px",
        "width": "250px",
        "background-color": "#0077B5",
        "border-radius": "10px",
        "margin-right": "50px",
        "display": "flex",
        "align-items": "center",  # Centraliza verticalmente
        "justify-content": "center",  # Centraliza horizontalmente
        "margin-left": "50px",
        "color": "white",
        "text-align": "center",
        "border": "2px solid black",
    })

    # Github
    card2 = html.Div([html.A(
            "Projeto no Git/Hub",
            href="https://github.com/murilocechin/imoveis_sp",  
            style={"color": "white", "text-decoration": "none","text-shadow": "1px 1px 2px black"},
            target="_blank"  
        )], style={
        "height": "60px",
        "width": "250px",
        "background-color": "#28a745",
        "border-radius": "10px",
        "margin-right": "50px",
        "display": "flex",
        "align-items": "center",  # Centraliza verticalmente
        "justify-content": "center",  # Centraliza horizontalmente
        "margin-left": "50px",
        "color": "white",
        "text-align": "center",
        "border": "2px solid black",
    })

    # Kaggle
    card3 = html.Div([html.A(
            "Base de Dados no Kaggle",
            href="https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019",  
            style={"color": "white", "text-decoration": "none","text-shadow": "1px 1px 2px black"},
            target="_blank"  
        )], style={
        "height": "60px",
        "width": "250px",
        "background-color": "#20BEFF",
        "border-radius": "10px",
        "margin-right": "50px",
        "display": "flex",
        "align-items": "center",  # Centraliza verticalmente
        "justify-content": "center",  # Centraliza horizontalmente
        "margin-left": "50px",
        "color": "white",
        "text-align": "center",
        "border": "2px solid black",
    })

    # Criando Quadro de descrição do projeto

    descricao = html.Div([
        html.H3(["Descrição do Projeto"], style={"text-aling": "center","margin-top":"20px"}),
        html.Blockquote(
            [
                html.P(
                    [
                        "O objetivo principal do projeto é realizar a predição dos preços de venda e aluguel de apartamentos na cidade de São Paulo, fazendo uso de uma base de dados didática disponível no site Kaggle. O processo abrange diversas etapas, começando pelo pré-processamento dos dados, seguido por uma análise exploratória minuciosa. Na sequência, ocorre a etapa de modelagem, onde algoritmos de machine learning são empregados para criar um modelo capaz de predizer os preços desejados."
                    ]
                ),
                html.P(
                    [
                        "Além disso, como parte integrante do projeto, foi desenvolvida uma interface gráfica que simplifica a interação do usuário. Nessa interface, o usuário fornece os parâmetros relevantes, e o algoritmo utiliza esses dados para retornar as predições dos preços de aluguel e venda. Essa abordagem permite uma utilização mais intuitiva e acessível do modelo desenvolvido, tornando-o mais amigável para usuários que não possuem expertise em machine learning."
                    ]
                ),
                html.P(
                    [
                        "Em resumo, o projeto compreende desde a manipulação inicial dos dados até a entrega de uma interface prática, proporcionando uma solução abrangente e eficaz para a predição de preços imobiliários em São Paulo."
                    ]
                ),
            ],
            style={"text-align": "left", "margin-right":"20px", "margin-left":"20px"},
        )                   
                           ], style={"height":"80%", "width":"80%", "background-color": "white","border-radius": "10px","border": "2px solid black", "margin": "auto","text-align": "center"})


    layout = html.Div([
        # Div Titulo
        html.Div([
        html.P("Projeto Imóveis SP", 
               style={"text-align": "center", "color": "white", "font-size": "22px", "font-weight": "bold", "margin": "auto"}),
    ], 
    style={'width': '100%', 'height': '100px', 'background-color': "#404040", 'display': 'flex', 'align-items': 'center'}),

    # Div da descrição
    html.Div([descricao], style={'background-color': "#0D6EFD", "width":"100%", "height":"500px",'justify-content': 'center', 'display': 'flex','align-items': 'center'}),

    # Div Link para redes
    html.Div([card1,card2,card3
    ], style={"background-color": "#404040", "width":"100%", "height":"200px",'justify-content': 'center', 'display': 'flex','align-items': 'center'})
    ])
    return layout


# Função para o layout da página "Predição"
# Lista de bairros que alimenta a escolha do usuario
def district_list():
    # Importando dataframe auxiliar
    df = pd.read_csv("modelagem/district_cat_aux.csv", index_col=0)

    # Criando lista de distritos
    lista_distritos = df.index.to_list()

    return lista_distritos


# Funções Auxiliares
def district_cat(district_name):
    """
    Função que converte bairro em categoria conforme documentação da analise exploratória.
    """
    # Importando dataframe auxiliar
    df = pd.read_csv("modelagem/district_cat_aux.csv", index_col=0) 

    return df.loc[ df.index==district_name, "class"].iloc[0]


def layout_predicao():

    # Definindo Layout para a pagina de predição 

    layout = html.Div([
        # Criando Div do texto do cabeçalho
        html.Div([
            html.P("Insira as caracteristicas do seu apartamento!", style={"text-align" : "center", "color": "white", "font-size":"18px", "font-weight": "bold"})
        ], style={'width': '100%', 'height': '60px', 'background-color': "#404040", 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

        # Criando cabeçalho
        html.Div([
            # Condominio
            html.Div([
                # Label
                html.Div(["Condominio"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Input(id="condominio", type="text", style={"max-width": "140px", "overflow": "hidden", 'margin-top': '10px'})],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px"})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Tamanho
            html.Div([
                # Label
                html.Div(["Tamanho"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Input(id="tamanho", type="text", style={"max-width": "140px", "overflow": "hidden", 'margin-top': '10px'})],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px"})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Quartos
            html.Div([
                # Label
                html.Div(["Quartos"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="quartos", options=[
                               {"label": "1", "value": 1},
                               {"label": "2", "value": 2},
                               {"label": "3", "value": 3},
                               {"label": "4+", "value": 4}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Banheiros
            html.Div([
                # Label
                html.Div(["Banheiros"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="banheiros", options=[
                               {"label": "1", "value": 1},
                               {"label": "2", "value": 2},
                               {"label": "3", "value": 3},
                               {"label": "4", "value": 4},
                               {"label": "5+", "value": 5}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Suites
            html.Div([
                # Label
                html.Div(["Suites"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="suites", options=[
                               {"label": "1", "value": 1},
                               {"label": "2", "value": 2},
                               {"label": "3", "value": 3},
                               {"label": "4+", "value": 4}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Vagas de Garagem
            html.Div([
                # Label
                html.Div(["Vagas de Garagem"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="vagas", options=[
                                {"label": "0", "value": 0},
                               {"label": "1", "value": 1},
                               {"label": "2", "value": 2},
                               {"label": "3", "value": 3},
                               {"label": "4+", "value": 4}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Elevador
            html.Div([
                # Label
                html.Div(["Elevador"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="elevador", options=[
                               {"label": "Sim", "value": 1},
                               {"label": "Não", "value": 0}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Mobilhado
            html.Div([
                # Label
                html.Div(["Mobilhado"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="mobilhado", options=[
                               {"label": "Sim", "value": 1},
                               {"label": "Não", "value": 0}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Pscina
            html.Div([
                # Label
                html.Div(["Pscina"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(id="pscina", options=[
                               {"label": "Sim", "value": 1},
                               {"label": "Não", "value": 0}
                           ], multi=False)],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Distrito
            html.Div([
                # Label
                html.Div(["Localização"], style={"width": "93%", "height": "30%", "background-color": "#292929", "margin": "5px","text-align" : "center", "color":"white"}),

                # Input
                html.Div([dcc.Dropdown(
                id="localizacao",
                options=district_list(),
                placeholder="Selecione o bairro",
                multi=False,
                clearable=True,
                style={"width": "93%", "margin": "5px", "margin-top": "10px"},
            )],
                          style={"width": "93%", "height": "40%", "background-color": "#292929", "margin": "5px",'margin-top': '10px'})

            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px"}),

            # Botão de calcular
            html.Div([
                dbc.Button('Calcular', id='submit-val', n_clicks=0)
            ], style={"width": "150px", "height": "70%", "background-color": "#292929", "margin": "10px", "border-radius": "10px","display":"flex",'justify-content': 'center', 'align-items': 'center'})



        ], style={'width': '100%', 'height': '170px', 'background-color': "white", "display":"flex", 'justify-content': 'center', 'align-items': 'center'}),
        
        # Valores Preditos
        html.Div([
            # Valor de predição rent
            html.Div([

                html.Div([], id="output-predict-rent",style={"height": "50%", "width":"55%", 'background-color': "white", 'align-items': 'center', "border-radius": "10px", "border": "6px solid #007BFF", 'justify-content': 'center', "display":"flex","font-size": "20px", "font-weight": "bold"})

            ], style={'background-color': "#404040", "height":"100%", 'width': '50%', "display":"flex", 'justify-content': 'center', 'align-items': 'center'}),

            # Valor de predição sale
            html.Div([

                html.Div([],id="output-predict-sale",style={"height": "50%", "width":"55%", 'background-color': "white", 'align-items': 'center', "border-radius": "10px", "border": "6px solid orange", 'justify-content': 'center', "display":"flex","font-size": "20px", "font-weight": "bold"})

            ], style={'background-color': "#404040", "height":"100%", 'width': '50%', "display":"flex", 'justify-content': 'center', 'align-items': 'center'}),


        ], style={'height': '250px', "display":"flex"})
    ])

    return layout


# Callback para salvar os dados em um arquivo CSV
@app.callback(
    [Output("output-predict-rent", "children"),Output("output-predict-sale", "children")],
    [Input('submit-val', 'n_clicks')],
    [dash.State('condominio', 'value'),
     dash.State('tamanho', 'value'),
     dash.State('quartos', 'value'),
     dash.State('banheiros', 'value'),
     dash.State('suites', 'value'),
     dash.State('vagas', 'value'),
     dash.State('elevador', 'value'),
     dash.State('mobilhado', 'value'),
     dash.State('pscina', 'value'),
     dash.State('localizacao', 'value')
     ]
)
def update_output(n_clicks, condo, size, rooms, toilets, suites, parking, elevator, furnished, pool, location):
    if n_clicks > 0:

        location = district_cat(location)

        # Ajuste de Variavel Dummy 
        if location == "A":
            district_B_return = 0
            district_C_return = 0
            district_D_return = 0
        elif location == "B":
            district_B_return = 1
            district_C_return = 0
            district_D_return = 0
        elif location == "C":
            district_B_return = 0
            district_C_return = 1
            district_D_return = 0
        elif location == "D":
            district_B_return = 0
            district_C_return = 0
            district_D_return = 1
        else:
            district_B_return = 0
            district_C_return = 0
            district_D_return = 0


        # Criar um dataframe com os dados de entrada
        input_data = pd.DataFrame([{
            "Condo": int(condo),
            "Size": int(size),
            "Rooms": rooms,
            "Toilets": toilets,
            "Suites": suites,
            "Parking": parking,
            "Elevator": elevator,
            "Furnished": furnished,
            "Swimming Pool": pool,
            "district_class_B": district_B_return,
            "district_class_C": district_C_return,
            "district_class_D": district_D_return,        
        }])


        # Fazer as predições
        rent_prediction = modelo_aluguel_final.predict(input_data)[0]
        sale_prediction = modelo_venda_final.predict(input_data)[0]

        # Exibir os resultados
        return f"Preço de Aluguel: R$ {rent_prediction:.2f}", f"Preço de Venda: R$ {sale_prediction:.2f}"

    return "", ""
    




# Função para o layout da página "Pré-Processamento"
def layout_preprocessamento():
    # Ler o conteúdo do arquivo HTML gerado a partir do Jupyter Notebook
    with open('pre-processamento/pre_processamento_doc.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Criar o layout da página usando o conteúdo HTML incorporado
    layout = html.Div(
        [html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '100%'})],
        className="content", style={'width': '100%', 'height': '800px'})

    return layout

# Função para o layout da página "Análise Exploratória"
def layout_analise_exploratoria():
    # Ler o conteúdo do arquivo HTML gerado a partir do Jupyter Notebook
    with open('analise-exploratoria/analise_exploratoria_doc.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Criar o layout da página usando o conteúdo HTML incorporado
    layout = html.Div(
        [html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '100%'})],
        className="content", style={'width': '100%', 'height': '800px'})

    return layout


# Função para o layout da página "Modelagem"
def layout_modelagem_rent():
    # Ler o conteúdo do arquivo HTML gerado a partir do Jupyter Notebook
    with open('modelagem/modelagem_rent_doc.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Criar o layout da página usando o conteúdo HTML incorporado
    layout = html.Div(
        [html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '100%'})],
        className="content", style={'width': '100%', 'height': '800px'})

    return layout

def layout_modelagem_sale():
    # Ler o conteúdo do arquivo HTML gerado a partir do Jupyter Notebook
    with open('modelagem/modelagem_sale_doc.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Criar o layout da página usando o conteúdo HTML incorporado
    layout = html.Div(
        [html.Iframe(srcDoc=html_content, style={'width': '100%', 'height': '100%'})],
        className="content", style={'width': '100%', 'height': '800px'})

    return layout

# Layout do aplicativo
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Início", href="/menu1", active=True, style={'margin-right': '20px', 'margin-top': '5px', 'margin-left': '5px'})),
                dbc.NavItem(dbc.NavLink("Predição", href="/menu2", active=True, style={'margin-right': '20px', 'margin-top': '5px', 'margin-left': '5px'})),
                dbc.NavItem(dbc.NavLink("Pré-Processamento", href="/menu3", active=True, style={'margin-right': '20px', 'margin-top': '5px', 'margin-left': '5px'})),
                dbc.NavItem(dbc.NavLink("Análise Exploratória", href="/menu4", active=True, style={'margin-right': '20px', 'margin-top': '5px', 'margin-left': '5px'})),
                dbc.DropdownMenu([dbc.DropdownMenuItem("Aluguel (Rent)", href="/menu5_1"), dbc.DropdownMenuItem("Venda (Sale)", href="/menu5_2")], color="primary", label="Modelagem",style={'margin-right': '20px', 'margin-top': '5px', 'margin-left': '5px'}, menu_variant="dark")
            ],
            pills=True,
            style={'height': '55px'}
        ),
        style={'backgroundColor': '#292929'}, 
    ),
    html.Div(id="content"),
])

# Callback para atualizar o conteúdo com base no item do menu selecionado
@app.callback(
    Output("content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/menu1":
        return layout_inicio()
    elif pathname == "/menu2":
        return layout_predicao()
    elif pathname == "/menu3":
        return layout_preprocessamento()
    elif pathname == "/menu4":
        return layout_analise_exploratoria()
    elif pathname == "/menu5_1":
        return layout_modelagem_rent()
    elif pathname == "/menu5_2":
        return layout_modelagem_sale()
    else:
        return layout_inicio()



# Executando o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
