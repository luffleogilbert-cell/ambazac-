# -*- coding: utf-8 -*-
import onecode as oc
from onecode import file_input, slider, file_output, Logger
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
import warnings

warnings.filterwarnings('ignore')


def run():
    # --- INPUTS INTERACTIFS ---
    Logger.info("Initialisation de l'interface...")
    fichier_geo = file_input('Données géochimiques', 'Points_geochimie_AMBAZAC.geojson')
    fichier_mnt = file_input('Image MNT', 'MNT_25M_AMBAZAC_IMAGE.tif')

    facteur_mad = slider('Facteur MAD (seuil anomalie)', 2.0, min=1.0, max=5.0)
    poids_au = slider('Poids Or (Au)', 0.40, min=0.0, max=1.0)
    poids_as = slider('Poids Arsenic (As)', 0.20, min=0.0, max=1.0)
    poids_w = slider('Poids Tungstène (W)', 0.20, min=0.0, max=1.0)
    poids_bi = slider('Poids Bismuth (Bi)', 0.20, min=0.0, max=1.0)

    # --- CHARGEMENT ---
    try:
        data = gpd.read_file(fichier_geo)
        Logger.info(f'Géochimie chargée : {len(data)} points')

        with rasterio.open(fichier_mnt) as src:
            mnt = src.read(1).astype(float)
            mnt[mnt == src.nodata] = np.nan
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        Logger.info('MNT chargé avec succès')
    except Exception as e:
        Logger.error(f"Erreur lors de la lecture des fichiers : {e}")
        return

    # --- ANALYSE DES ANOMALIES ---
    elements_carte = ['Au_ppb', 'As_ppm', 'W_ppm', 'Sn_ppm']
    cmaps = ['YlOrRd', 'PuRd', 'Blues', 'Greens']
    fig1, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for i, (elem, cmap) in enumerate(zip(elements_carte, cmaps)):
        if elem in data.columns:
            serie = data[elem]
            seuil = serie.median() + facteur_mad * (serie - serie.median()).abs().median()
            masque = serie > seuil

            ax = axes[i]
            norm = mcolors.LogNorm(vmin=serie[serie > 0].min(), vmax=serie.max())
            ax.scatter(data.X[~masque], data.Y[~masque], c=serie[~masque], cmap=cmap, norm=norm, s=10, alpha=0.4)
            ax.scatter(data.X[masque], data.Y[masque], c=serie[masque], cmap=cmap, norm=norm, s=40, edgecolors='black',
                       zorder=5)
            ax.set_title(f'{elem} ({masque.sum()} anomalies)')

    plt.tight_layout()
    oc.plot(fig1, title="Cartographie des Anomalies")
    fig1.savefig(file_output('Carte_Anomalies', 'anomalies.png'))
    plt.close(fig1)

    # --- SCORE DE POTENTIEL ---
    def norm_log(serie):
        s = np.log10(serie.replace(0, np.nan))
        return (s - s.min()) / (s.max() - s.min())

    score = (norm_log(data['Au_ppb']) * poids_au +
             norm_log(data['As_ppm']) * poids_as +
             norm_log(data['W_ppm']) * poids_w +
             norm_log(data['Bi_ppm']) * poids_bi).fillna(0)

    top = data[score > score.quantile(0.95)]

    # --- CARTE FINALE ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    # Calcul rapide Hillshade pour le fond
    dz_dy, dz_dx = np.gradient(mnt, 25, 25)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    asp = np.arctan2(dz_dy, -dz_dx)
    hs = np.clip(
        np.cos(np.radians(45)) * np.cos(slope) + np.sin(np.radians(45)) * np.sin(slope) * np.cos(np.radians(135) - asp),
        0, 1)

    ax2.imshow(hs, cmap='gray', extent=extent, origin='upper')
    ax2.imshow(mnt, cmap='terrain', extent=extent, origin='upper', alpha=0.3)

    sc = ax2.scatter(data.X, data.Y, c=score, cmap='hot_r', s=20, alpha=0.8)
    ax2.scatter(top.X, top.Y, c='cyan', s=60, marker='*', edgecolors='black', label='Top 5% Potentiel')

    plt.colorbar(sc, ax=ax2, label='Score de potentiel')
    ax2.legend()
    ax2.set_title("Carte de Synthèse du Potentiel Minéral")

    oc.plot(fig2, title="Potentiel Minéral")
    fig2.savefig(file_output('Carte_Potentiel', 'potentiel.png'))
    plt.close(fig2)

    Logger.info("Analyse terminée. Résultats disponibles dans l'onglet Output.")


if __name__ == "__main__":
    run()
