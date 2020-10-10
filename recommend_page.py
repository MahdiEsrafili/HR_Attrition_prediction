import dash
import dash_table
import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df[:10].to_dict('records'),
)

if __name__ == '__main__':
    app.run_server(debug=True)