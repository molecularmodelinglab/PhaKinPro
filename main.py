from flask import Flask, render_template, request, abort, Response, jsonify
from smiles import get_molecule_data_from_smiles
from ZincRX.zinc_rx import MODEL_DICT
from csv_smiles import get_csv_from_smiles

app = Flask(__name__)


@app.route('/')
def home():
    print("main")
    return render_template('index.html')


@app.route('/models', methods=['GET'])
def mol_properties():
    print("model_selection")
    return jsonify(list(MODEL_DICT.keys())), 200


@app.route('/smiles', methods=['POST'])
def smiles():
    print("smiles")

    data = get_molecule_data_from_smiles(request.json.get('smiles'), request.json.get('options'))

    if data is None:
        return abort(400)

    return data


@app.route('/smiles-csv', methods=['POST'])
def smiles_csv():
    print("csv")

    smiles = request.json.get('smiles')
    options = request.json.get('options')

    num_models = sum([val for key, val in options.items() if key not in ["calculate_ad", "make_prop_img"]])

    if len(smiles) * num_models > 100:
        return abort(413)

    csv = get_csv_from_smiles(request.json.get('smiles'), request.json.get('options'))

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment;"}
    )
