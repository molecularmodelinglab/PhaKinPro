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

MODEL_DICT = {
    'Hepatic Stability 15min': 'Dataset_01B_hepatic-stability_15min_imbalanced-morgan_RF.pgz',
    'Hepatic Stability 30min': 'Dataset_01C_hepatic-stability_30min_imbalanced-morgan_RF.pgz',
    'Hepatic Stability 60min': 'Dataset_01D_hepatic-stability_60min_imbalanced-morgan_RF.pgz',
    'Microsomal Half-life Sub-cellular': 'Dataset_02A_microsomal-half-life-subcellular_imbalanced-morgan_RF.pgz',
    'Microsomal Half-life 30min': 'Dataset_02B_microsomal-half-life_30-min_binary_unbalanced_morgan_RF.pgz',
    'Renal Clearance 0.1': 'dataset_03_renal-clearance_0.1-threshold_balanced-morgan_RF.pgz',
    'Renal Clearance 0.5': 'dataset_03_renal-clearance_0.5-threshold_imbalanced-morgan_RF.pgz',
    'Renal Clearance 1.0': 'dataset_03_renal-clearance_1.0-threshold_balanced-morgan_RF.pgz',
    'BBB Permeability': 'dataset_04_bbb-permeability_balanced-morgan_RF.pgz',
    'CNS Activity': 'dataset_04_cns-activity_1464-compounds_imbalanced-morgan_RF.pgz',
    'CACO2': 'Dataset_05A_CACO2_binary_unbalanced_morgan_RF.pgz',
    'Plasma Protein Binding': 'Dataset_06_plasma-protein-binding_binary_unbalanced_morgan_RF.pgz',
    'Plasma Half-life 12hr': 'Dataset_08_plasma_half_life_12_hr_balanced-morgan_RF.pgz',
    'Plasma Half-life 1hr': 'Dataset_08_plasma_half_life_1_hr_balanced-morgan_RF.pgz',
    'Plasma Half-life 6hr': 'Dataset_08_plasma_half_life_6_hr_imbalanced-morgan_RF.pgz',
    'Microsomal Intrinsic Clearance': 'Dataset_09_microsomal-intrinsic-clearance_12uL-min-mg-threshold-imbalanced-morgan_RF.pgz',
    'Oral Bioavailability 0.5': 'dataset_10_oral_bioavailability_0.5_threshold_imbalanced-morgan_RF.pgz',
    'Oral Bioavailability 0.8': 'dataset_10_oral_bioavailability_0.8_balanced-morgan_RF.pgz'
}


MODEL_DICT_INVERT = {val: key for key, val in MODEL_DICT.items()}

CLASS_DICT = {
    0: "Inactive",
    1: "Active"
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


def main(smiles, calculate_ad=True, make_prop_img=False, **kwargs):
    def default(key, d):
        if key in d.keys():
            return d[key]
        else:
            return False

    models = [f for f in glob.glob("./ZincRX/models/*.pgz")]
    models_data = [f for f in glob.glob("./ZincRX/models/*.pbz2")]

    values = []

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

        values.append([MODEL_DICT_INVERT[os.path.basename(model_endpoint)], CLASS_DICT[int(pred)], str(round(float(pred_proba)*100, 2))+"%", AD_DICT[ad], svg_str])

    return values
