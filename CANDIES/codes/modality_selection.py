from sklearn.metrics import adjusted_rand_score
from esda.moran import Moran
from libpysal.weights import DistanceBand
from libpysal.weights import KNN
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def normalize(value, min_value, max_value, invert=False):
    if invert:
        return 1 - ((value - min_value) / (max_value - min_value))
    else:
        return (value - min_value) / (max_value - min_value)

def modality_selection(modality1_cluster,modality2_cluster,modality1_embs=None,modality2_embs=None,spatial_coor=None,ground_truth=None):

    if ground_truth is not None:
        ari_score1 = adjusted_rand_score(ground_truth, modality1_cluster)
        ari_score2 = adjusted_rand_score(ground_truth, modality2_cluster)
        if ari_score1 > ari_score2:
            print('We recommend you use modality 1 as condition in the denoise phase!')
        else:
            print('We recommend you use modality 2 as condition in the denoise phase!')

    if ground_truth is None:

        modality1_features = modality1_embs
        modality1_labels = modality1_cluster

        protein_features = modality2_embs
        protein_labels = modality2_cluster


        protein_silhouette = silhouette_score(protein_features, protein_labels)
        rna_silhouette = silhouette_score(modality1_features, modality1_labels)

        protein_dbi = davies_bouldin_score(protein_features, protein_labels)
        rna_dbi = davies_bouldin_score(modality1_features, modality1_labels)

        protein_ch = calinski_harabasz_score(protein_features, protein_labels)
        rna_ch = calinski_harabasz_score(modality1_features, modality1_labels)
        
        coords = np.array(spatial_coor)[:, :2]  # Use only the first two columns
        labels = modality1_cluster.astype(float)  # Ensure labels are numerical
        w_knn = KNN(coords, k=3)
        moran = Moran(labels, w_knn)
        rna_moran = moran.I

        coords = np.array(spatial_coor)[:, :2]  # Use only the first two columns
        labels = modality2_cluster.astype(float)  # Ensure labels are numerical
        w_knn = KNN(coords, k=3)
        moran1 = Moran(labels, w_knn)
        pro_moran = moran1.I
        
        min_values = {
            "Silhouette": -1, 
            "DBI": 0,  
            "CH Index": 0, 
            "Moran": -1 
        }

        max_values = {
            "Silhouette": 1, 
            "DBI": 10,
            "CH Index": 200,
            "Moran": 1 
        }

        protein_silhouette_norm = normalize(protein_silhouette, min_values["Silhouette"], max_values["Silhouette"])
        rna_silhouette_norm = normalize(rna_silhouette, min_values["Silhouette"], max_values["Silhouette"])

        protein_dbi_norm = normalize(protein_dbi, min_values["DBI"], max_values["DBI"], invert=True)
        rna_dbi_norm = normalize(rna_dbi, min_values["DBI"], max_values["DBI"], invert=True)

        protein_ch_norm = normalize(protein_ch, min_values["CH Index"], max_values["CH Index"])
        rna_ch_norm = normalize(rna_ch, min_values["CH Index"], max_values["CH Index"])

        protein_moran_norm = normalize(pro_moran, min_values["Moran"], max_values["Moran"])
        rna_moran_norm = normalize(rna_moran, min_values["Moran"], max_values["Moran"])

        weights = {
            "Silhouette": 0.25,
            "DBI": 0.25,
            "CH Index": 0.25,
            "Moran": 0.25
        }

        weighted_protein = (protein_silhouette_norm * weights["Silhouette"] +
                            protein_dbi_norm * weights["DBI"] +
                            protein_ch_norm * weights["CH Index"] +
                            protein_moran_norm * weights["Moran"])

        weighted_rna = (rna_silhouette_norm * weights["Silhouette"] +
                        rna_dbi_norm * weights["DBI"] +
                        rna_ch_norm * weights["CH Index"] +
                        rna_moran_norm * weights["Moran"])

        if weighted_rna > weighted_protein:
            print('We recommend you use modality 1 as condition in the denoise phase!')
        else:
            print('We recommend you use modality 2 as condition in the denoise phase!')

