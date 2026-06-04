import dash
from dash import Dash, dash_table, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
from collections import defaultdict
import base64
import io
import os
import json

app = dash.Dash(__name__)

# Sample data to display in the graph
import pandas as pd

entity_types = ['None', 'LOC', 'ORG', 'MISC', 'PER']

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload File', id='upload', className='button'),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # # Allow multiple files to be uploaded
        # multiple=True
    ),
    dash_table.DataTable(
        id='upload_output',
        columns=[],
        style_header={
            'text-align': 'center',
            'background-color': '#555555',
            'color': 'white'
        },
        data=[],
        style_cell={
            'textAlign': 'left',
            'minWidth': '100px', 'width': '100px', 'maxWidth': '150px',
            'whiteSpace': 'nowrap',  # Prevent text from wrapping
            'overflow': 'hidden',  # Hide text that can't fit in the cell
            'textOverflow': 'ellipsis'  # Use ellipsis for overflowed text
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Long Column Name'}, 'textAlign': 'left'},
        ],
        # Removed style_data as style_cell now includes the necessary properties
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        column_selectable="single",
        row_selectable='multi',
        row_deletable=True,
        selected_rows=[],
        page_action='native',
        page_current=0,
        page_size=5,
    ),
    dcc.Store(id='stored-data'),

    html.Div(id='instance-container', children=[
        html.Div([
            html.H3('Label Color Map:'),
            html.Div(id='instance_label_map'),
        ],
            style={'display': 'flex', 'align-items': 'center'},
        ),
        html.Div([
            html.H3('Example:'),
            html.Div(id='instance_sentence',
                     style={'padding': '10px', 'margin-right': '10px',
                            'direction': 'rtl', 'unicode-bidi': 'embed'}),
        ],
            style={'display': 'flex', 'align-items': 'center'}
        ),
    ]),

    dcc.Dropdown(
        id='example-id-selector',
        options=[{'label': str(i), 'value': i} for i in range(101)],  # IDs from 0 to 100
        value=0,  # Default value
        style={'width': '50%'}
    ),

    # Button to retrieve the selected example
    html.Button('Retrieve Example', id='retrieve-example', n_clicks=0, className='button'),

    html.Div([
        html.Div([
            html.H3('Mistakes:', style={'text-align': 'center'}),
            dcc.Dropdown(
                id='mistakes-breakdown',
                options=[],
                placeholder='Mistake Breakdown',
                multi=True,
            ),
            dcc.Dropdown(
                id='mistakes-type',
                options=[],
                # Update these options as needed
                placeholder='Mistake Type',
            ),
            # dcc.Dropdown(
            #     id='type-span',
            #     options=[],
            #     placeholder='Mistake Span',
            #     multi=True
            # ),
            # html.Button('Submit Mistakes', id='submit-mistakes', n_clicks=0, className='button'),
            # dcc.Input(id='mistakes-count', type='number', value=0, style={'width': '100%', 'margin': '10px 0'}),
            # dcc.Textarea(id='mistakes-breakdown', placeholder='Describe the mistakes here',
            #              style={'width': '100%', 'height': '50px', 'margin': '10px 0'}),
            # dcc.Textarea(id='mistake-types', placeholder='Specify mistake types here',
            #              style={'width': '100%', 'height': '50px', 'margin': '10px 0'}),
        ], style={'padding': '20px', 'margin': '10px', 'flex': 1, 'background-color': '#f7f7f7',
                  'border-radius': '5px'}),

        html.Div([
            html.H3('Missings:', style={'text-align': 'center'}),
            dcc.Dropdown(
                id='missing-breakdown',
                options=[],
                placeholder='Missing Breakdown',
                multi=True,
            ),
            dcc.Dropdown(
                id='missing-type',
                options=[],
                # Update these options as needed
                placeholder='Missing Type',
            ),
            # dcc.Input(id='missings-count', type='number', value=0, style={'width': '100%', 'margin': '10px 0'}),
            # dcc.Textarea(id='missings-breakdown', placeholder='Describe the missings here',
            #              style={'width': '100%', 'height': '50px', 'margin': '10px 0'}),
            # dcc.Textarea(id='missing-types', placeholder='Specify missing types here',
            #              style={'width': '100%', 'height': '50px', 'margin': '10px 0'}),
        ], style={'padding': '20px', 'margin': '10px', 'flex': 1, 'background-color': '#f7f7f7',
                  'border-radius': '5px'}),
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        html.Button('Submit Example', id='submit-example', n_clicks=0, className='button'),
        html.Button('Delete Example', id='delete-example', n_clicks=0, className='button')
        # Add this line for the delete button
    ], style={'display': 'flex', 'justifyContent': 'center'}, className='buttons-container'),
    html.Div([
        html.Button('View Annotation Table', id='view-table', n_clicks=0, className='button'),
        # Add this line for the delete button
    ], style={'display': 'flex', 'justifyContent': 'center'}, className='buttons-container'),

    dash_table.DataTable(
        id='annotations-table',
        columns=[
            {'name': 'Example ID', 'id': 'example_id'},
            {'name': 'Mistakes', 'id': 'mistakes'},
            {'name': 'Mistake Breakdown', 'id': 'mistake_breakdown'},
            {'name': 'Mistake Types', 'id': 'mistake_types'},
            {'name': 'Missings', 'id': 'missings'},
            {'name': 'Missing Breakdown', 'id': 'missing_breakdown'},
            {'name': 'Missing Types', 'id': 'missing_types'},
            {'name': 'Message ID', 'id': 'message_id'},
            {'name': 'Text', 'id': 'text'},
            {'name': 'Truncated', 'id': 'truncated'},
            {'name': 'Truncation Amount', 'id': 'truncation_amount'},
        ],
        data=[],
        style_header={
            'text-align': 'center'
        },
        style_cell={
            'textAlign': 'left',
            'minWidth': '100px', 'width': '100px', 'maxWidth': '150px',
            'whiteSpace': 'nowrap',  # Prevent text from wrapping
            'overflow': 'hidden',  # Hide text that can't fit in the cell
            'textOverflow': 'ellipsis'  # Use ellipsis for overflowed text
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Long Column Name'}, 'textAlign': 'left'},
        ],
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        column_selectable="single",
        row_selectable='multi',
        row_deletable=True,
        selected_rows=[],
        page_action='native',
        page_current=0,
        page_size=100,
        # ... (style and other properties)
    ),
    dcc.Store(id='annotations-store'),

])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'jsonl' in filename:
            # Assume that the user uploaded a JSON Lines file
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')), lines=True)
        else:
            return html.Div([
                'Unsupported file type!'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df  # Return the processed DataFrame


def color_examples(sentence, extractions):
    color_map = {
        'LOC': 'darkgreen', 'PER': 'deepskyblue',
        'ORG': 'darkcyan', 'MISC': 'palevioletred',
    }
    label_color_map = [html.Span(lb, style={'background-color': color_map[lb], 'margin-right': '5px', 'padding': '2px',
                                            'color': 'white'}) for lb, color in color_map.items()]

    # Initial index to keep track of where to start the next uncolored segment
    last_idx = 0
    colored_sentence = []  # This will store the colored and uncolored segments

    for entity, entity_group, start, end in extractions:
        # Add the text before the entity (uncolored)
        if start > last_idx:
            colored_sentence.append(sentence[last_idx:start])

        # Add the entity text (colored)
        colored_sentence.append(html.Span(sentence[start:end], style={
            'background-color': color_map.get(entity_group, 'grey'),
            'margin-right': '5px',
            'padding': '2px',
            'color': 'white'
        }))

        last_idx = end  # Update the last index to the end of the current entity

    # Add any remaining text after the last entity (uncolored)
    if last_idx < len(sentence):
        colored_sentence.append(sentence[last_idx:])

    return label_color_map, colored_sentence


@app.callback(
    [
        Output("upload_output", "data"),
        Output("upload_output", "columns"),
        Output("stored-data", "data"),
    ],
    [
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    ]
)
def upload_file(file_content, file_name):
    if file_content is not None:
        df = parse_contents(file_content, file_name)
        df = df.reset_index()

        return df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns], df.to_dict('records')

    return [], [], []


@app.callback(
    [
        Output("instance_label_map", "children"),
        Output("instance_sentence", "children"),
        Output("mistakes-breakdown", "options"),
        Output("mistakes-type", "options"),
        Output("mistakes-breakdown", "value"),  # Reset value of mistakes-breakdown dropdown
        Output("mistakes-type", "value"),  # Reset value of mistakes-type dropdown
        Output("missing-breakdown", "options"),
        Output("missing-type", "options"),
        Output("missing-breakdown", "value"),  # Reset value of missing-breakdown dropdown
        Output("missing-type", "value"),  # Reset value of missing-type dropdown
    ],
    [
        Input('retrieve-example', 'n_clicks'),
        State("example-id-selector", "value"),
        State("stored-data", "data")
    ]
)
def display_selected_example(n_clicks, selected_id, stored_data):
    if n_clicks > 0 and stored_data and 0 <= selected_id < len(stored_data):
        sentence = stored_data[selected_id]['preprocessedText']
        extractions = eval(stored_data[selected_id]['extractions'])
        if sentence:
            label_color_map, colored_sentence = color_examples(sentence, extractions)
            words = [f'{w}*@#*{i}' for i, w in enumerate(sentence.split())]
        else:
            label_color_map = ''
            colored_sentence = ''
            words = []

        mistake_breakdown = []
        # type_span = []
        # Loop through each tuple and create the dropdown options
        for i, extraction in enumerate(extractions):
            span, entity, start, end = extraction
            # mistake_breakdown.append({'label': span, 'value': f'{span}*@#*{entity}'})
            mistake_breakdown.append(f'{span}*@#*{entity}*@#*{i}')
        error_types = ['Entity Type', 'Entity Boundary', 'Truncation', 'Tokenization']
        return label_color_map, colored_sentence, \
               mistake_breakdown, error_types, None, None, \
              words, error_types, None, None

    return [[], [], [], [], [], [], [], [], [], []]


def read_annotations(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []


def write_annotations(file_path, annotations):
    with open(file_path, 'w') as file:
        json.dump(annotations, file, ensure_ascii=False, indent=4)


def process_annotation(data, example_id, mistakes_breakdown, mistakes_type, missing_breakdown, missing_type):
    # Create a defaultdict for mistake types
    mistake_types = defaultdict(list)
    # Check if mistakes_breakdown is None
    if mistakes_breakdown is None:
        mistakes_breakdown = []
    # Check if mistakes_type is None
    if mistakes_type is None:
        mistakes_type = ''  # Replace with a default type or an empty string

    for span in mistakes_breakdown:
        mistake_types[mistakes_type].append(span)

    # Create a defaultdict for missing types
    missing_types = defaultdict(list)
    # Check if missing_breakdown is None
    if missing_breakdown is None:
        missing_breakdown = []
    # Check if missing_type is None
    if missing_type is None:
        missing_type = ''  # Replace with a default type or an empty string

    for missing in missing_breakdown:
        missing_types[missing_type].append(missing)

    # Construct the new annotation dictionary
    new_annotation = {
        'example_id': example_id,
        'mistakes': len(mistakes_breakdown),
        'mistake_breakdown': json.dumps(mistakes_breakdown),
        'mistake_types': json.dumps(dict(mistake_types)),
        'missings': len(missing_breakdown),
        'missing_breakdown': json.dumps(missing_breakdown),
        'missing_types': json.dumps(dict(missing_types)),
        'message_id': data[example_id]['message_id'],
        'text': data[example_id]['preprocessedText'],
        # Add any additional fields needed for the annotation
    }

    return new_annotation



@app.callback(
    [Output("annotations-table", "data"), Output('annotations-store', 'data')],
    [Input('submit-example', 'n_clicks'), Input('delete-example', 'n_clicks'), Input('view-table', 'n_clicks')],
    [State('example-id-selector', 'value'), State('upload_output', 'data'),
     State('mistakes-breakdown', 'value'), State('mistakes-type', 'value'),
     State('missing-breakdown', 'value'), State('missing-type', 'value'), State('upload-data', 'filename')]
)
def submit_annotation(submit, delete, view_table, example_id, data, mistakes_breakdown, mistakes_type,
                      missing_breakdown, missing_type, uploaded_filename):
    if not uploaded_filename:
        return [], []

    annotations_dir = 'annotation_outputs'
    os.makedirs(annotations_dir, exist_ok=True)
    file_path = os.path.join(annotations_dir, f"{os.path.splitext(uploaded_filename)[0]}_annotations.json")

    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'view-table':
        return read_annotations(file_path), read_annotations(file_path)

    stored_annotations = read_annotations(file_path)

    if button_id == 'submit-example' and data and example_id is not None:
        new_annotation = process_annotation(data, example_id, mistakes_breakdown, mistakes_type, missing_breakdown,
                                            missing_type)
        stored_annotations.append(new_annotation)

        # Remove duplicates and sort by example_id
        unique_annotations = {ann['example_id']: ann for ann in stored_annotations}
        stored_annotations = sorted(unique_annotations.values(), key=lambda x: x['example_id'])

    if button_id == 'delete-example':
        stored_annotations = [ann for ann in stored_annotations if ann['example_id'] != example_id]

    write_annotations(file_path, stored_annotations)

    return stored_annotations, stored_annotations



# Step 5: Run the App
if __name__ == '__main__':
    app.run_server(debug=True)
