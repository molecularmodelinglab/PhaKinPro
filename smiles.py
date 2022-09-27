from rdkit.Chem import MolFromSmiles, Draw
from ZincRX.zinc_rx import main

# to set the coloring display of the outcomes on the website
COLORS = {
    "Hepatic stability <= 50% at 15 minutes": "red",
    "Hepatic stability <= 50% between 15 and 30 minutes": "red",
    "Hepatic stability <= 50% between 30 and 60 minutes": "orange",
    "Hepatic stability > 50% at 60 minutes": "balck",
    "Sub-cellular Hepatic Half-life > 30 minutes": "red",
    "Sub-cellular Hepatic Half-life <= 30 minutes": "black",
    "Tissue Hepatic Half-life > 30 minutes": "red",
    "Tissue Hepatic Half-life <= 30 minutes": "black",
    "Renal clearance below 0.10 ml/min/kg": "red",
    "Renal clearance between 0.10 and 0.50 ml/min/kg": "red",
    "Renal clearance between 0.50 and 1.00 ml/min/kg": "orange",
    "Renal clearance above 1.00 ml/min/kg": "black",
    "Does not permeate blood brain barrier": "black",
    "Does permeate blood brain barrier": "red",
    "Does not exhibit central nervous system activity": "black",
    "Does exhibit central nervous system activity": "red",
    "Does not permeate Caco-2": "black",
    "Does permeate Caco-2": "red",
    "Plasma protein binding": "red",
    "Weak/non plasma protein binding": "black",
    "Half-life below 1 hour": "red",
    "Half-life between 1 and 6 hours": "red",
    "Half-life between 6 and 12 hours": "orange",
    "Half-life above 12 hours": "balck",
    "Microsomal intrinsic clearance < 12 uL/min/mg": "black",
    "Microsomal intrinsic clearance >= 12 uL/min/mg": "red",
    "Less and 0.5 F": "red",
    "Between 0.5 and 0.8 F": "orange",
    "Above 0.8 F": "black",
    "Inconsistent result: no prediction": "red"
}


def get_molecule_data_from_smiles(smiles_str, options):
    molecule = MolFromSmiles(smiles_str)

    if molecule is None:
        return None

    data = main(smiles_str, **options)

    # add color coding of text
    data = [_ + [COLORS[_[1]]] for _ in data]

    # skip if no models selected
    if len(data) == 0:
        return None

    return {
        'svg': Draw.MolsToGridImage([molecule], useSVG=True, molsPerRow=1),
        'SMILES': smiles_str,
        'pred_data': data,
    }
