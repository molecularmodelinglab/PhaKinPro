'''
    Wrapper class for the tensorflow models. This class will follow scikit-learn's API.
'''
import os

# Libraries
import tensorflow as tf
import numpy as np
import keras

from rdkit.Chem import MolFromSmiles, MolToSmiles, PandasTools
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs


def get_path(path):
    return os.path.join(os.path.dirname(__file__), path)


# TF custom initializer
@keras.saving.register_keras_serializable()
class CustomInitializer(tf.keras.initializers.Initializer):
    '''
        Bias initializer to give custom biases into each class.
    '''

    def __init__(self, bias: tf.Tensor):
        self.custom_bias = bias

    def __call__(self, shape=None, dtype=None, **kwargs):
        return self.custom_bias

    def get_config(self):
        base_config = super().get_config()
        config = {'custom_bias': keras.saving.serialize_keras_object(self.custom_bias)}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        custom_config = config.pop('custom_bias')
        custom_bias = keras.saving.deserialize_keras_object(custom_config)
        return cls(custom_bias, **config)


class PhaKinProCYP:
    def __init__(self, cyp_type='3A4', endpoint="inh", method='unanimous'):
        '''
            Initialize the PhaKinProCYP model.

            args:
                cyp_type: str
                    The CYP enzyme to be predicted: "3A4", "2D6" or "2C9". Default is '3A4'.
                method: str
                    The method to be used to predict the CYP enzyme: "unanimous" or "majority". Default is 'unanimous'.

            returns:
                None
        '''
        self.cyp_type = cyp_type
        self.endpoint = endpoint
        self.method = method

        if self.cyp_type == '3A4':
            if self.endpoint == 'inh':
                self.models = [  # NPV=0.91, PPV=0.65
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_32_16_8.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_64_32_16.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_128_64_32.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_256_128_64.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_512_128_64.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_512_256_64.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_1024_256_64.keras')),
                    keras.models.load_model(get_path('models/CYP3A4_inh_MLP_cw_ob_1024_512_64.keras')),
                ]
                self.thresholds = [0.5] * len(self.models)
            elif self.endpoint == 'sub':
                self.models = [
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_8_4.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_16_8_4.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_32_16_16_8.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_64_8_4.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_64_16_8.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_128_64_64_64_64_32.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_256_128_128_128_128_128_64.keras')),
                    keras.models.load_model(get_path(f'models/CYP3A4_sub_MLP_ob_512_128_128_128_128_128_64.keras')),
                ]
                self.thresholds = [0.5] * len(self.models)
            else:
                raise ValueError(f"Unrecognized endpoint {self.endpoint}")

        elif self.cyp_type == '2D6':
            if self.endpoint == 'inh':
                self.models = [  # NPV=0.91, PPV=0.65
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_32_16_8.keras')),
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_64_32_16.keras')),
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_128_64_32.keras')),
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_512_128_64.keras')),
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_1024_256_64.keras')),
                    keras.models.load_model(get_path('models/CYP2D6_inh_MLP_cw_ob_1024_512_64.keras')),
                ]
                self.thresholds = [0.5] * len(self.models)
            elif self.endpoint == 'sub':
                self.models = [  # NPV=0.87, PPV=0.71
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_64_16_16_16_8.keras')),
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_64_32_32_32_32_16.keras')),
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_128_64_64_32.keras')),
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_256_128_64.keras')),
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_256_128_128_128_128_128_64.keras')),
                    keras.models.load_model(get_path(f'models/CYP2D6_sub_MLP_ob_512_128_64.keras')),
                ]
                self.thresholds = [0.5] * len(self.models)
            else:
                raise ValueError(f"Unrecognized endpoint {self.endpoint}")

        elif self.cyp_type == '2C9':
            if self.endpoint == 'inh':
                self.models = [  # NPV=0.90, PPV=0.60
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_32_16_8.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_64_32_16.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_128_64_32.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_256_128_64.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_512_128_64.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_512_256_64.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_1024_256_64.keras')),
                    keras.models.load_model(get_path('models/CYP2C9_inh_MLP_cw_ob_1024_512_64.keras')),
                ]
                self.thresholds = [0.6] * len(self.models)
            elif self.endpoint == 'sub':
                self.models = [  # NPV=0.87, PPV=0.72
                    keras.models.load_model(get_path(f'models/CYP2C9_sub_MLP_ob_64_16_8.keras')),
                    keras.models.load_model(get_path(f'models/CYP2C9_sub_MLP_ob_128_64_32.keras')),
                    keras.models.load_model(get_path(f'models/CYP2C9_sub_MLP_ob_256_128_128_64.keras')),
                ]
                self.thresholds = [0.6, 0.65, 0.85]
            else:
                raise ValueError(f"Unrecognized endpoint {self.endpoint}")
        else:
            raise ValueError(f"Unrecognized CYP {self.cyp_type}")

    def predict(self, molecules):
        '''
            Predict the CYP enzyme for a list of smiles.

            args:
                molecules as list of smiles or rdkit mol objects

            returns:
                np.array: predicted labels
        '''
        # Convert smiles to rdkit mol objects
        if isinstance(molecules[0], str):
            molecules = [MolFromSmiles(mol) for mol in molecules]

        # Generate fingerprints
        fps = [self._generate_fp(mol) for mol in molecules]

        # Predict
        if self.method == 'unanimous':
            return self._predict_ensemble_unanimous(self.models, fps, self.thresholds)[0]
        elif self.method == 'majority':
            raise NotImplementedError('Majority voting is not implemented yet.')
        else:
            raise ValueError('Invalid method. Choose between "unanimous" and "majority".')

    def predict_proba(self, molecules):
        '''
            Predict the CYP enzyme for a list of smiles.

            args:
                molecules as list of smiles or rdkit mol objects

            returns:
                np.array: predicted probabilities
        '''
        # Convert smiles to rdkit mol objects
        if isinstance(molecules[0], str):
            molecules = [MolFromSmiles(mol) for mol in molecules]

        # Generate fingerprints
        fps = [self._generate_fp(mol) for mol in molecules]

        # Predict
        if self.method == 'unanimous':
            return self._predict_ensemble_unanimous(self.models, fps, self.thresholds)[1]
        elif self.method == 'majority':
            raise NotImplementedError('Majority voting is not implemented yet.')
        else:
            raise ValueError('Invalid method. Choose between "unanimous" and "majority".')

    def _generate_fp(self, mol, fp_type='bit'):
        '''
            Generate the fingerprint of a molecule.

            args:
                mol: rdkit mol object
                    The molecule to be fingerprinted.
                type: str
                    The type of fingerprint to be generated: 'bit' or 'count'. Default is 'bit'.

            returns:
                fingerprint as rdkit bit vector
        '''
        if fp_type == 'bit':
            return SimilarityMaps.GetMorganFingerprint(mol, radius=3, nBits=2048, useChirality=True, fpType='bv')
        elif fp_type == 'count':
            return SimilarityMaps.GetMorganFingerprint(mol, radius=3, nBits=2048, useChirality=True, fpType='count')

    def _apply_applicability_domain(self, fp):
        '''
            Calculate Tanimoto similarity between the fingerprints of the molecule and the training set.

            args:
                fp:
                    The fingerprint of the molecule.

            returns:
                maximum Tanimoto similarity
        '''
        # Load the training set
        cyp2d6 = PandasTools.LoadSDF('additional_data/CYP2D6_subs_combined_curated_cls.sdf')['ROMol'].to_list()
        cyp2c9 = PandasTools.LoadSDF('additional_data/CYP2C9_subs_combined_curated_cls.sdf')['ROMol'].to_list()
        cyp3a4 = PandasTools.LoadSDF('additional_data/CYP3A4_subs_combined_curated_cls.sdf')['ROMol'].to_list()
        cyp2d6 = [MolToSmiles(mol) for mol in cyp2d6]
        cyp2c9 = [MolToSmiles(mol) for mol in cyp2c9]
        cyp3a4 = [MolToSmiles(mol) for mol in cyp3a4]
        train_smiles = list(set(cyp2d6).intersection(set(cyp2c9)).intersection(set(cyp3a4)))
        train_mols = [MolFromSmiles(smi) for smi in train_smiles]
        train_fps = [self._generate_fp(mol) for mol in train_mols]
        tani_sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        return max(tani_sims)

    def _predict_with_threshold(self, model: keras.Model, x_test: np.array, threshold: float):
        '''
            Predict the labels using a given model and probability threshold.

            Args:
                model: keras model to use for prediction.
                x_test: np.array containing the dataset to predict.
                threshold: probability threshold to use for prediction.

            Returns:
                np.array: predicted labels.
                np.array: predicted probabilities.

            Notes: This function will return an extra label
                for samples with a probability below the threshold.
                These samples are rejected.
        '''
        # predict_with_threshold
        probs = model.predict(tf.convert_to_tensor(x_test), verbose=0, batch_size=256)
        probs = np.hstack((probs, np.full((len(probs), 1), threshold)))
        preds = np.argmax(probs, axis=1)
        probs = np.max(probs, axis=1)
        return preds, probs

    def _predict_ensemble_unanimous(self, models: list, x_test: np.array, thresholds: list):
        '''
            Predict the labels for a given dataset using an ensemble
            of models. This model tries to maximize Precision for both classes
            by removing samples that are predicted to be negative by at least
            one model.

            Args:
                models: list of keras models to use for prediction.
                x_test: np.array containing the dataset to predict.
                thresholds: list of probability thresholds to use for prediction.

            Returns:
                np.array: predicted labels.
                np.array: predicted probabilities.
        '''
        # predict with each model
        preds = []
        probs = []
        for i, model in enumerate(models):
            pred, prob = self._predict_with_threshold(model, x_test, thresholds[i])
            preds.append(pred)
            probs.append(prob)
        preds_std = np.std(preds, axis=0)
        preds = np.mean(preds, axis=0)
        preds = np.array([pred if std == 0 else 2 for pred, std in zip(preds, preds_std)])
        preds = preds.astype(int)
        probs = np.mean(probs, axis=0)
        return preds, probs