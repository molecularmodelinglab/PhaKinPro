import csv
from io import StringIO

import glob
import gzip
import bz2
import os
import _pickle as cPickle

import numpy as np
from rdkit import DataStructs, Chem
from rdkit.Chem import MolFromSmiles, AllChem
from tqdm import tqdm

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

file = "C:\\Users\\James\\Downloads\\frdb-v2023-02-15\\frdb\\frdb-pk.tsv"

import pandas as pd
df = pd.read_csv(file, delimiter="\t")

f_unbond = df[["pk_analyte_smiles","pk_funbound_value"]].copy()
hl = df[["pk_analyte_smiles", "pk_thalf_units", "pk_thalf_value"]].copy()

f_unbond.dropna(inplace=True)
hl.dropna(inplace=True)

time_convert = {"day": 24, "s": 1/3600, "min": 1/60, "h": 1}
converted_times = [row["pk_thalf_value"] * time_convert[row["pk_thalf_units"]] for i, row in hl.iterrows()]
hl = pd.DataFrame({"pk_analyte_smiles": hl["pk_analyte_smiles"], "pk_thalf_value": converted_times})

hl["pk_thalf_value"] = hl["pk_thalf_value"].astype(float)
f_unbond["pk_funbound_value"] = f_unbond["pk_funbound_value"].astype(float)

hl_g_mean = hl.groupby("pk_analyte_smiles").mean()
hl_g_std = hl.groupby("pk_analyte_smiles").std().fillna(0)
hl = hl_g_mean.join(hl_g_std, rsuffix="_std")
hl["pk_analyte_smiles"] = hl_g_mean.index

f_g_mean = f_unbond.groupby("pk_analyte_smiles").mean()
f_g_std = f_unbond.groupby("pk_analyte_smiles").std().fillna(0)
f_unbond = f_g_mean.join(f_g_std, rsuffix="_std")
f_unbond["pk_analyte_smiles"] = f_g_mean.index


def label_hl(val):
    if val < 1:
        return 3
    if val < 6:
        return 2
    if val < 12:
        return 1
    return 0


def label_f(val):
    if val > 60:
        return 1
    else:
        return 0


#### HL

hl["label1"] = (hl["pk_thalf_value"] < 1)
hl["label2"] = (hl["pk_thalf_value"] < 6)
hl["label3"] = (hl["pk_thalf_value"] < 12)
model1 = cPickle.load(gzip.open("C:\\Users\\James\\OneDrive\\TropshaLab\\Website\\PhaKinPro\\PhaKinPro\\models\\Dataset_01B_hepatic-stability_15min_imbalanced-morgan_RF.pgz", 'rb'))
model2 = cPickle.load(gzip.open("C:\\Users\\James\\OneDrive\\TropshaLab\\Website\\PhaKinPro\\PhaKinPro\\models\\Dataset_01C_hepatic-stability_30min_imbalanced-morgan_RF.pgz", 'rb'))
model3 = cPickle.load(gzip.open("C:\\Users\\James\\OneDrive\\TropshaLab\\Website\\PhaKinPro\\PhaKinPro\\models\\Dataset_01D_hepatic-stability_60min_imbalanced-morgan_RF.pgz", 'rb'))

res1 = []
res2 = []
res3 = []

for smi in tqdm(hl["pk_analyte_smiles"].to_list()):
    fp = np.zeros((2048, 1))
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        res1.append([smi, None, None])
        res2.append([smi, None, None])
        res3.append([smi, None, None])
        continue
    _fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
    DataStructs.ConvertToNumpyArray(_fp, fp)
    fp = fp.reshape(1, -1)

    pred1 = model1.predict(fp)
    pred2 = model2.predict(fp)
    pred3 = model3.predict(fp)

    prob1 = model1.predict_proba(fp)
    prob2 = model2.predict_proba(fp)
    prob3 = model3.predict_proba(fp)

    res1.append([pred1, prob1])
    res2.append([pred2, prob2])
    res3.append([pred3, prob3])

hl["pred1"] = [_[0] for _ in res1]
hl["pred2"] = [_[0] for _ in res2]
hl["pred3"] = [_[0] for _ in res3]

    # average_prob = np.mean(np.max([prob1, prob2, prob3],axis=0))

    # if pred1 == 0:
    #     if pred2 == 1 or pred3 == 1:
    #         pred = -1
    #     else:
    #         pred = 0
    # elif pred2 == 0:
    #     if pred3 == 1:
    #         pred = -1
    #     else:
    #         pred = 1
    # elif pred3 == 0:
    #     pred = 2
    # else:
    #     pred = 3
    # res.append([smi, pred, average_prob])














# from phakinpro import main
#
# def get_csv_from_smiles(smiles_list, options):
#     # CSV writer expects a file object, not a string.
#     # StringIO can be used to store a string as a file-like object.
#
#     options["make_prop_img"] = False  # do not need to create images for csv
#
#     headers = [key for key, val in options.items() if ((key not in ["calculate_ad", "make_prop_img"]) and val)]
#
#     if options["calculate_ad"]:
#         headers = headers + [_+"_AD" for _ in headers]
#
#     string_file = StringIO()
#     writer = csv.DictWriter(string_file, fieldnames=['SMILES', *headers])
#     writer.writeheader()
#
#     for smiles in tqdm(smiles_list):
#         molecule = MolFromSmiles(smiles)
#
#         row = {'SMILES': smiles}
#
#         if molecule is None:
#             row['SMILES'] = f"(invalid){smiles}"
#             writer.writerow(row)
#             continue
#
#         data = main(smiles, **options)
#
#         for model_name, pred, pred_proba, ad, _ in data:
#             try:
#                 pred_proba = float(pred_proba[:-1]) / 100  # covert back to 0-1 float
#                 row[model_name] = pred_proba if pred == 1 else 1 - pred_proba  # this is to make sure its proba for class 1
#             except ValueError:
#                 row[model_name] = "No prediction"  # if pred_proba is string skip
#             if options["calculate_ad"]:
#                 row[model_name + "_AD"] = ad
#
#         writer.writerow(row)
#
#     return string_file.getvalue()
#
#
# hl_results = get_csv_from_smiles(hl["pk_analyte_smiles"].to_list(), options={"Plasma Half-life": True, "calculate_ad": True})
# f_results = get_csv_from_smiles(f_unbond["pk_analyte_smiles"].to_list(), options={"Plasma Protein Binding": True, "calculate_ad": True})
