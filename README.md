# TCM prescription generation
Traditional Chinese medicine (TCM) prescription generation with graph AI model
## Introduction

This repository contains the model part of our paper, which includes the graph embedding layer and herb recommendation layer. Among them,  Node2Vec is used to capture the latent features of each herb from the  herb-ingredient-target network.  The Herb Recommendation layer consists of the MLP structure that fit the embedding of each TCM formula to an exact score to assess the quality.

## Installation
The graph model is implemented based on [PYG](https://github.com/pyg-team/pytorch_geometric) , which requires the pytorch version >= 1.12.0. More information about PYG installation can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The installation command as follows.

```
pip install torch==1.13.0+cu11x --extra-index-url https://download.pytorch.org/whl/cu11x
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu11x.html
```

## Data preparation

The graph data is constructed with Node.csv and Edge.csv
| Node_id |                      Nodes                      | Label |
| :-----: | :---------------------------------------------: | :---: |
|    0    | 4-aminobutyrate aminotransferase, mitochondrial |   2   |
|    1    |              sophoraisoflavanone,a              |   1   |
|    2    |           Euphorbiae Humifusae Herba            |   0   |

| source_Node_id | target_Node_id |
| :------------: | :------------: |
|      697       |      2031      |
|      7145      |      741       |
|      1426      |      5967      |

The training data for the Rocommender is as follows, which is generated based on the real TCM prescription

|                          Compounds                           | Score |
| :----------------------------------------------------------: | :---: |
| Crotonis Fructus, Arecae Semen, Aucklandiae Radix, Citri Reticulatae Pericarpium Viride, Olibanun | 0.29  |
| Arum Ternatum Thunb., Magnolia Officinalis Rehd Et Wils, Coptidis Rhizoma, Phragmitis Rhizomam, Acoritataninowii Rhizoma, Gardeniae Fructus | 0.86  |
| Arum Ternatum Thunb., Clematidis Armandii Caulis, Aconiti Lateralis Radix Praeparata | 0.75  |
| Folium Artemisiae Argyi, Phellodendri Chinrnsis Cortex, Zingiber Officinale Roscoe |  0.6  |
| Aconitum Kusnezoffii Reichb, licorice, Zingiberis Rhizoma, Magnolia Officinalis Rehd Et Wils |   1   |
|    Scutellariae Radix, Curcumae Radix, Gardeniae Fructus     | 0.38  |

## Results

We inferred all potential prescriptions for the specific disease with our model.  The recommended formulas are displayed in the following table, which are selected for the biological experiment.
|                         TCM Formulas                         | Score  |
| :----------------------------------------------------------: | :----: |
| Commelinae Herba,Paeoniae Radix Alba | 0.9683 |
| Asteris Radix Et Rhizoma,Andrographis Herba | 0.9679 |
| Atractylodes Macrocephala Koidz.,Paeoniae Radix Alba | 0.966 |
| Scutellariae Radix,Paeoniae Radix Alba | 0.9592 |
| Coptidis Rhizoma,Paeoniae Radix Alba | 0.9559 |

## Project Support

This material is based upon work supported by PengCheng Lab.

