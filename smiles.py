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


AD_MEANING = {
    "Hepatic stability <= 50% at 15 minutes": "Red means supports < 15 minutes green supports > 15 minutes",
    "Hepatic stability <= 50% between 15 and 30 minutes": "Red means supports < 15 or > 30 minutes green supports 15-30 minutes",
    "Hepatic stability <= 50% between 30 and 60 minutes": "Red means supports < 30 or > 60 minutes green supports 30-60 minutes",
    "Hepatic stability > 50% at 60 minutes": "Red means supports < 60 minutes green supports > 60 minutes",
    "Sub-cellular Hepatic Half-life > 30 minutes": "Red means supports > 30 minutes green supports < 60 minutes",
    "Sub-cellular Hepatic Half-life <= 30 minutes": "Red means supports > 30 minutes green supports < 60 minutes",
    "Tissue Hepatic Half-life > 30 minutes": "Red means supports > 30 minutes green supports < 60 minutes",
    "Tissue Hepatic Half-life <= 30 minutes": "Red means supports > 30 minutes green supports < 60 minutes",
    "Renal clearance below 0.10 ml/min/kg": "Red means supports < 0.10 green supports > 0.10",
    "Renal clearance between 0.10 and 0.50 ml/min/kg": "Red means supports < 0.10 or > 0.50 green supports 0.10 - 0.50",
    "Renal clearance between 0.50 and 1.00 ml/min/kg": "Red means supports < 0.50 or > 1.00 green supports 0.50 - 1.00",
    "Renal clearance above 1.00 ml/min/kg": "Red means supports < 1.00 green supports > 1.00",
    "Does not permeate blood brain barrier": "Red means supports no permeation green supports permeation",
    "Does permeate blood brain barrier": "Red means supports no permeation green supports permeation",
    "Does not exhibit central nervous system activity": "Red means supports no NCS activity green supports NCS activity",
    "Does exhibit central nervous system activity": "Red means supports no NCS activity green supports NCS activity",
    "Does not permeate Caco-2": "Red means supports no permeation green supports permeation",
    "Does permeate Caco-2": "Red means supports no permeation green supports permeation",
    "Plasma protein binding": "Red means supports no binding green supports binding",
    "Weak/non plasma protein binding": "Red means supports no binding green supports binding",
    "Half-life below 1 hour": "Red means supports < 1 hr green supports > 1 hr",
    "Half-life between 1 and 6 hours": "Red means supports < 1 hr or > 6 hrs green supports 1 - 6 hrs",
    "Half-life between 6 and 12 hours": "Red means supports < 6 hr or > 12 hrs green supports 6 - 12 hrs",
    "Half-life above 12 hours": "Red means supports < 12 hrs green supports > 12 hrs",
    "Microsomal intrinsic clearance < 12 uL/min/mg": "Red means supports < 12 and green supports > 12",
    "Microsomal intrinsic clearance >= 12 uL/min/mg": "Red means supports < 12 and green supports > 12",
    "Less and 0.5 F": "Red means supports < 0.5F green supports > 0.5F",
    "Between 0.5 and 0.8 F": "Red means supports < 0.5F or > 0.8F green supports 0.5 - 0.8F",
    "Above 0.8 F": "Red means supports < 0.8F green supports > 0.8F",
    "Inconsistent result: no prediction": ""
}

color_text = False  # set to True if you want to color code the text


def get_molecule_data_from_smiles(smiles_str, options):
    molecule = MolFromSmiles(smiles_str)

    if molecule is None:
        return None

    data = main(smiles_str, **options)

    # add color coding of text
    if color_text:
        data = [_ + [COLORS[_[1]]] for _ in data]
    else:
        data = [_ + ["black"] for _ in data]

    print(options)

    if "make_prop_img" in options.keys():
        data = [_ + [AD_MEANING[_[1]]] for _ in data]
    else:
        data = [_ + [""] for _ in data]

    # skip if no models selected
    if len(data) == 0:
        return None

    return {
        'svg': Draw.MolsToGridImage([molecule], useSVG=True, molsPerRow=1),
        'SMILES': smiles_str,
        'pred_data': data,
    }
