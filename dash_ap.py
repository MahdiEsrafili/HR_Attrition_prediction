import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import joblib
import numpy as np
import pandas as pd 
from ModifiedLabelEncoder import ModifiedLabelEncoder
from Recommendation import Recommend
import flask
flask_app = flask.Flask(__name__)
model_pipe = joblib.load('model_pipe.joblib')
le_pipe = joblib.load('le_pipe.joblib')

app = dash.Dash(__name__, server = flask_app, title= 'Personel Attrition')
datax = pd.read_csv('data/sample.csv')
indx = 18
sample = datax.iloc[indx:indx+1]
main_data = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# data.set_index('EmployeeNumber', inplace = True)

numeric_fields = ['Age', 'DailyRate', 
       'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
       'StockOptionLevel', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']
categorical_fields = ['BusinessTravel', 'Department', 'EducationField',
       'JobRole', 'MaritalStatus', 'OverTime']

all_columns = numeric_fields + categorical_fields + ['Attrition']
main_data = main_data[all_columns].reindex(sorted(all_columns), axis=1)
labeled_data = main_data
main_data = le_pipe.transform(main_data)
recommender = Recommend(main_data, 'Attrition')
style = {
        'height': '25px',
        'width':'200px',
        'margin-top': '10px',
        'margin-bottom': '10px',
        'margin-right': '10px',
        'margin-left': '0px',
       }

style_numeric = {
        'height': '25px',
        'width':'50px',
        'margin-top': '10px',
        'margin-bottom': '10px',
        'margin-right': '10px',
        'margin-left': '5px',
       }


app.layout = html.Div( [

     html.Div([html.Label([field,
         dcc.Input(id ="{}_id".format(field), value=sample.loc[indx, field], type= 'number',placeholder="input {}".format(field),  style=style_numeric ) 
     ]) for field in numeric_fields]),
     html.Br(),

     html.Label(["BusinessTravel", dcc.Dropdown(id="BusinessTravel_id", options = [
         {'label':'Travel_Rarely', 'value':'Travel_Rarely'},
         {'label':'Travel_Frequently', 'value':'Travel_Frequently'},
         {'label':'Non-Travel', 'value':'Non-Travel'}
    #  ], value=['Travel_Rarely'] , style=style
     ], value=sample.loc[indx, 'BusinessTravel'] , style=style
      )]),


    html.Br(),
     html.Label(["Department", dcc.Dropdown(id="Department_id", options = [
         {'label':'Sales', 'value':'Sales'},
         {'label':'Research & Development', 'value':'Research & Development'},
         {'label':'Human Resources', 'value':'Human Resources'}
    #  ], value='Research & Development',style=style)]),
     ], value=sample.loc[indx, 'Department'],style=style)]),


    html.Br(),
    html.Label(["EducationField", dcc.Dropdown(id="EducationField_id", options = [
         {'label':'Life Sciences', 'value':'Life Sciences'},
         {'label':'Medical', 'value':'Medical'},
         {'label':'Marketing', 'value':'Marketing'},
         {'label':'Technical Degree', 'value':'Technical Degree'},
         {'label':'Human Resources', 'value':'Human Resources'},
         {'label':'Other', 'value':'Other'}
    #  ], value='Medical',style=style)]),
     ], value=sample.loc[indx, 'EducationField'],style=style)]),


    # html.Br(),
    # html.Label(["Gender", dcc.Dropdown(id="Gender_id", options = [
    #      {'label':'Female', 'value':'Female'},
    #      {'label':'Male', 'value':'Male'}
    #  ], value='Female',style=style)]),
   

    html.Br(),
    html.Label(["JobRole", dcc.Dropdown(id="JobRole_id", options = [
         {'label':'Sales Executive', 'value':'Sales Executive'},
         {'label':'Research Scientist', 'value':'Research Scientist'},
         {'label':'Laboratory Technician', 'value':'Laboratory Technician'},
         {'label':'Manufacturing Director', 'value':'Manufacturing Director'},
         {'label':'Healthcare Representative', 'value':'Healthcare Representative'},
         {'label':'Manager', 'value':'Manager'},
         {'label':'Sales Representative', 'value':'Sales Representative'},
         {'label':'Research Director', 'value':'Research Director'},
         {'label':'Human Resources', 'value':'Human Resources'}
    #  ], value='Sales Executive',style=style)]),
     ], value=sample.loc[indx, 'JobRole'],style=style)]),


    html.Br(),
    html.Label(["MaritalStatus", dcc.Dropdown(id="MaritalStatus_id", options = [
         {'label':'Single', 'value':'Single'},
         {'label':'Married', 'value':'Married'},
         {'label':'Divorced', 'value':'Divorced'}
    #  ], value='Single',style=style)]),
     ], value=sample.loc[indx, 'MaritalStatus'],style=style)]),

    # html.Br(),
    # html.Label(["Over18", dcc.Dropdown(id="Over18_id", options = [
    #      {'label':'Y', 'value':'Y'}
    #  ], value='Y',style=style)]),

     
    html.Br(),
    html.Label(["OverTime", dcc.Dropdown(id="OverTime_id", options = [
         {'label':'Yes', 'value':'Yes'},
         {'label':'No', 'value':'No'}
    #  ], value='Yes',style=style)]),
     ], value=sample.loc[indx, 'OverTime'],style=style)]),


    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0, style= {'background-color':'MediumSeaGreen'}),
    html.Br(),
    html.Div(id='Attrition')  ,
     html.Br(),
      html.Br(),
     html.Label("Person Info"),
    dash_table.DataTable(
    id='person_table',
    columns = [],
    data= [],
     ),
     html.Br(),

     html.Label("Similar Persons with NO Attrition"),

     html.Br(),

     dash_table.DataTable(
    id='recommend_table',
    columns = [],
    data = [],
     ),
])

i = 1
@app.callback([Output("Attrition", "children"),
     Output('person_table', 'data'),
     Output('person_table', 'columns'),
     Output('recommend_table', 'data'),
     Output('recommend_table', 'columns')],
    [Input('submit-val', 'n_clicks')],
    [State("{}_id".format(field), 'value') for field in numeric_fields + categorical_fields]
    )
def submit(n_clicks, *vals):
    clmn =  numeric_fields + categorical_fields
    df = pd.DataFrame({clmn[v] : [vals[v]] for v in range(len(vals))})
    df = df.reindex(sorted(df.columns), axis=1)
    x = le_pipe.transform(df)
    attrition = model_pipe.predict(x)[0]
    attrition_list = ['No', 'Yes']
    columns = [{"name": i, "id": i} for i in df.columns] 
    all_columns_for_table = [{"name": i, "id": i} for i in sorted(all_columns)]
    recommend_data = recommender.recommend(x)
    recommend_data = le_pipe.inverse_transform(recommend_data)
    return f'Attrition: {attrition_list[attrition]}', df.to_dict('records'), columns, recommend_data.to_dict('records'), all_columns_for_table
    
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0',debug=False, port= 5000)