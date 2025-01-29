from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from scipy.spatial.distance import cdist
import numpy as np

import glob
import gzip
import bz2
import os
import _pickle as cPickle

import io
import matplotlib.pyplot as plt

from cpy_model import PhaKinProCYP

# god hates me so in my version of python I cannot supress these damn user warning so I do this nuclear option instead
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

MODEL_DICT = {
    'Hepatic Stability': ['Dataset_01B_hepatic-stability_15min_imbalanced-morgan_RF.pgz',
                          'Dataset_01C_hepatic-stability_30min_imbalanced-morgan_RF.pgz',
                          'Dataset_01D_hepatic-stability_60min_imbalanced-morgan_RF.pgz'],
    'Microsomal Half-life Sub-cellular': ['Dataset_02A_microsomal-half-life-subcellular_imbalanced-morgan_RF.pgz'],
    'Microsomal Half-life Tissue': ['Dataset_02B_microsomal-half-life_30-min_binary_unbalanced_morgan_RF.pgz'],
    'Renal Clearance': ['dataset_03_renal-clearance_0.1-threshold_balanced-morgan_RF.pgz',
                        'dataset_03_renal-clearance_0.5-threshold_imbalanced-morgan_RF.pgz',
                        'dataset_03_renal-clearance_1.0-threshold_balanced-morgan_RF.pgz'],
    'BBB Permeability': ['dataset_04_bbb-permeability_balanced-morgan_RF.pgz'],
    'CNS Activity': ['dataset_04_cns-activity_1464-compounds_imbalanced-morgan_RF.pgz'],
    'CACO2': ['Dataset_05A_CACO2_binary_unbalanced_morgan_RF.pgz'],
    'Plasma Protein Binding': ['Dataset_06_plasma-protein-binding_binary_unbalanced_morgan_RF.pgz'],
    'Plasma Half-life': ['Dataset_08_plasma_half_life_12_hr_balanced-morgan_RF.pgz',
                         'Dataset_08_plasma_half_life_1_hr_balanced-morgan_RF.pgz',
                         'Dataset_08_plasma_half_life_6_hr_imbalanced-morgan_RF.pgz'],
    'Microsomal Intrinsic Clearance': ['Dataset_09_microsomal-intrinsic-clearance_12uL-min-mg-threshold-imbalanced-morgan_RF.pgz'],
    'Oral Bioavailability': ['dataset_10_oral_bioavailability_0.5_threshold_imbalanced-morgan_RF.pgz',
                             'dataset_10_oral_bioavailability_0.8_balanced-morgan_RF.pgz'],
    "CYP3A4": PhaKinProCYP(cyp_type="3A4", endpoint),
    "CYP2D6": PhaKinProCYP(cyp_type="2D6"),
    "CYP2C9": PhaKinProCYP(cyp_type="2C9")
}



# lol I'm just like screw code readability sorry
MODEL_DICT_INVERT = {v: key for key, val in MODEL_DICT.items() for v in val}

CLASSIFICATION_DICT = {
    'Hepatic Stability': {
        1: "Hepatic stability <= 50% at 15 minutes",
        2: "Hepatic stability <= 50% between 15 and 30 minutes",
        3: "Hepatic stability <= 50% between 30 and 60 minutes",
        4: "Hepatic stability > 50% at 60 minutes"
    },
    'Microsomal Half-life Sub-cellular': {
        0: "Sub-cellular Hepatic Half-life > 30 minutes",
        1: "Sub-cellular Hepatic Half-life <= 30 minutes"
    },
    'Microsomal Half-life Tissue': {
        0: "Tissue Hepatic Half-life > 30 minutes",
        1: "Tissue Hepatic Half-life <= 30 minutes"
    },
    'Renal Clearance': {
        1: "Renal clearance below 0.10 ml/min/kg",
        2: "Renal clearance between 0.10 and 0.50 ml/min/kg",
        3: "Renal clearance between 0.50 and 1.00 ml/min/kg",
        4: "Renal clearance above 1.00 ml/min/kg"
    },
    'BBB Permeability': {
        0: "Does not permeate blood brain barrier",
        1: "Does permeate blood brain barrier"
    },
    'CNS Activity': {
        0: "Does not exhibit central nervous system activity",
        1: "Does exhibit central nervous system activity"
    },
    'CACO2': {
        0: "Does not permeate Caco-2",
        1: "Does permeate Caco-2"
    },
    'Plasma Protein Binding': {
        0: "Plasma protein binder",
        1: "Weak/non plasma protein binder"
    },
    'Plasma Half-life': {
        1: "Half-life below 1 hour",
        2: "Half-life between 1 and 6 hours",
        3: "Half-life between 6 and 12 hours",
        4: "Half-life above 12 hours"
    },
    'Microsomal Intrinsic Clearance': {
        0: "Microsomal intrinsic clearance < 12 uL/min/mg",
        1: "Microsomal intrinsic clearance >= 12 uL/min/mg"
    },
    'Oral Bioavailability': {
        1: "Less than 0.5 F",
        2: "Between 0.5 and 0.8 F",
        3: "Above 0.8 F"
    },
    "CYP3A4 Inhibition": {
        0: "No inhibition",
        1: "Inhibition",
        2: "Inconsistent result: no prediction"
    },
    "CYP3A4 Substrate": {
        0: "Not a substrate",
        1: "Substrate",
        2: "Inconsistent result: no prediction"
    },
    "CYP2D6": PhaKinProCYP(cyp_type="2D6"),
    "CYP2C9": PhaKinProCYP(cyp_type="2C9")
}


AD_DICT = {
    True: "Inside",
    False: "Outside"
}


def run_prediction(model, model_data, smiles, calculate_ad=True):
    fp = np.zeros((2048, 1))
    _fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=3, nBits=2048)
    DataStructs.ConvertToNumpyArray(_fp, fp)

    pred_proba = model.predict_proba(fp.reshape(1, -1))[:, 1]
    pred = 1 if pred_proba > 0.5 else 0

    if pred == 0:
        pred_proba = 1-pred_proba

    if calculate_ad:
        ad = model_data["D_cutoff"] > np.min(cdist(model_data['Descriptors'].to_numpy(), fp.reshape(1, -1)))
        return pred, pred_proba, ad
    return pred, pred_proba, None


def get_prob_map(model, smiles):
    def get_fp(mol, idx):
        fps = np.zeros((2048, 1))
        _fps = SimilarityMaps.GetMorganFingerprint(mol, idx, radius=3, nBits=2048)
        DataStructs.ConvertToNumpyArray(_fps, fps)
        return fps

    def get_proba(fps):
        return float(model.predict_proba(fps.reshape(1, -1))[:, 1])

    mol = Chem.MolFromSmiles(smiles)
    fig, _ = SimilarityMaps.GetSimilarityMapForModel(mol, get_fp, get_proba)
    imgdata = io.StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data
    plt.savefig(imgdata, format="svg", bbox_inches="tight")

    return imgdata.getvalue()


def multiclass_ranking(ordered_preds):
    idx = 0
    one_detected = False
    for i, o in enumerate(ordered_preds):
        if int(o) == 1:
            if not one_detected:
                idx = i+1
                one_detected = True
        if int(o) == 0:
            if one_detected:
                idx = 0
                return idx
    return idx if idx != 0 else len(ordered_preds)+1


def main(smiles, calculate_ad=True, make_prop_img=False, **kwargs):

    # print(smiles)
    # print(os.getcwd())

    def default(key, d):
        if key in d.keys():
            return d[key]
        else:
            return False

    models = sorted([f for f in glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), "./models/*.pgz"))], key=lambda x: x.split("_")[1])
    models_data = sorted([f for f in glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),"./models/*.pbz2"))], key=lambda x: x.split("_")[1])

    values = {}

    for model_endpoint, model_data in zip(models, models_data):
        if not default(MODEL_DICT_INVERT[os.path.basename(model_endpoint)], kwargs):
            continue
        with gzip.open(model_endpoint, 'rb') as f:
            model = cPickle.load(f)

        with bz2.BZ2File(model_data, 'rb') as f:
            model_data = cPickle.load(f)

        pred, pred_proba, ad = run_prediction(model, model_data, smiles, calculate_ad=calculate_ad)

        svg_str = ""
        if make_prop_img:
            svg_str = get_prob_map(model, smiles)

        values.setdefault(MODEL_DICT_INVERT[os.path.basename(model_endpoint)], []).append([int(pred), str(round(float(pred_proba) * 100, 2)) + "%", AD_DICT[ad], svg_str])

    processed_results = []
    for key, val in values.items():
        if key in ['Hepatic Stability', 'Renal Clearance', 'Plasma Half-life', 'Oral Bioavailability']:
            new_pred = multiclass_ranking([_[0] for _ in val])
            if new_pred == 0:
                processed_results.append([key, "Inconsistent result: no prediction", "Very unconfident", "NA", ""])
            else:
                # this is because of how the hierarchical model works
                if new_pred in [1, 2]:
                    p = 0
                else:
                    p = new_pred - 2
                processed_results.append([key, CLASSIFICATION_DICT[key][new_pred], val[p][1], val[p][2], val[p][3]])
        else:
            processed_results.append([key, CLASSIFICATION_DICT[key][val[0][0]], val[0][1], val[0][2], val[0][3]])

    return processed_results


def write_csv_file(smiles_list, calculate_ad=False):
    headers = []
    for _key in MODEL_DICT.keys():
        headers.append(_key)
        headers.append(_key+"_proba")
        if calculate_ad:
            headers.append(_key+"_AD")

    string_file = StringIO()
    writer = csv.DictWriter(string_file, fieldnames=['SMILES', *headers])
    writer.writeheader()

    for smiles in tqdm(smiles_list):
        molecule = MolFromSmiles(smiles)

        row = {'SMILES': smiles}

        if molecule is None:
            row['SMILES'] = f"(invalid){smiles}"
            writer.writerow(row)
            continue

        data = main(smiles, calculate_ad=calculate_ad, **MODEL_DICT)

        for model_name, pred, pred_proba, ad, _ in data:
            print(pred, pred_proba, ad)
            try:
                pred_proba = float(pred_proba[:-1]) / 100  # covert back to 0-1 float
                row[model_name] = pred
                row[model_name + "_proba"] = pred_proba if pred == 1 else 1 - pred_proba  # this is to make sure its proba for class 1
            except ValueError:
                row[model_name] = "No prediction"  # if pred_proba is string skip
            if calculate_ad:
                row[model_name + "_AD"] = ad

        writer.writerow(row)

    return string_file.getvalue()


if __name__ == "__main__":
    import argparse
    import csv
    from io import StringIO
    from rdkit.Chem import MolFromSmiles
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="location to csv of SMILES")
    parser.add_argument("--outfile", type=str, default=os.path.join(os.getcwd(), "phakin_output.csv"),
                        help="location and file name for output")
    parser.add_argument("--smiles_col", type=str, default="SMILES",
                        help="column name containing SMILES of interest"),
    parser.add_argument("--ad", action="store_true",
                        help="calculate the AD")
