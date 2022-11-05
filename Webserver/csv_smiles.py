import csv
from io import StringIO
from rdkit.Chem import MolFromSmiles
from PhaKinPro.phakinpro import main


def get_csv_from_smiles(smiles_list, options):
    # CSV writer expects a file object, not a string. 
    # StringIO can be used to store a string as a file-like object.

    options["make_prop_img"] = False  # do not need to create images for csv

    headers = [key for key, val in options.items() if ((key not in ["calculate_ad", "make_prop_img"]) and val)]

    if options["calculate_ad"]:
        headers = headers + [_+"_AD" for _ in headers]

    string_file = StringIO()
    writer = csv.DictWriter(string_file, fieldnames=['SMILES', *headers])
    writer.writeheader()

    for smiles in smiles_list:
        molecule = MolFromSmiles(smiles)

        row = {'SMILES': smiles}

        if molecule is None:
            row['SMILES'] = f"(invalid){smiles}"
            writer.writerow(row)
            continue

        data = main(smiles, **options)

        for model_name, pred, pred_proba, ad, _ in data:
            try:
                pred_proba = float(pred_proba[:-1]) / 100  # covert back to 0-1 float
                row[model_name] = pred_proba if pred == 1 else 1 - pred_proba  # this is to make sure its proba for class 1
            except ValueError:
                row[model_name] = "No prediction"  # if pred_proba is string skip
            if options["calculate_ad"]:
                row[model_name + "_AD"] = ad

        writer.writerow(row)

    return string_file.getvalue()
