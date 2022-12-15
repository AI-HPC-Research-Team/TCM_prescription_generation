# TCM prescription generation
Traditional Chinese medicine (TCM) prescription generation with graph AI model
## Introduction

This repository contains the model part of our paper, which includes the graph embedding layer and herb recommendation layer. Among them,  Node2Vec is used to capture the latent features of each herb from the  herb-ingredient-target network.  The Herb Recommendation layer consists of the MLP structure that fit the embedding of each TCM formula to an exact score to assess the quality.

<img src="/Images/model_architecture.jpg" width="1100" height="650"/><br/>

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

We inferred all potential prescriptions for the specific disease with our model.  The top-10 formulas are displayed in the following table, which are selected for the biological experiment.

|                         TCM Formulas                         | Score  |
| :----------------------------------------------------------: | :----: |
| Euphorbiae Humifusae Herba,Scutellariae Radix,Polyporus Umbellatus(Pers)Fr.,Paeoniae Radix Alba | 0.9968 |
| Hedysarum Multijugum Maxim.,Euphorbiae Humifusae Herba,Polyporus Umbellatus(Pers)Fr.,Paeoniae Radix Alba | 0.9966 |
| Fraxini Cortex,Mori Follum,Scutellariae Radix,Paeoniae Radix Alba | 0.9966 |
| Euphorbiae Humifusae Herba,Polyporus Umbellatus(Pers)Fr.,Fructussophorae,Paeoniae Radix Alba | 0.9964 |
| Fraxini Cortex,Euphorbiae Humifusae Herba,Polyporus Umbellatus(Pers)Fr.,Paeoniae Radix Alba | 0.9962 |
|        Fraxini Cortex,Mori Follum,Paeoniae Radix Alba        | 0.9961 |
| Bistortae Rhizoma,Euphorbiae Humifusae Herba,Kochiae Fructus,Paeoniae Radix Alba | 0.996  |
| Euphorbiae Humifusae Herba,Portulacae Herba,Polyporus Umbellatus(Pers)Fr.,Paeoniae Radix Alba | 0.9959 |
| Fraxini Cortex,Scutellariae Radix,Evodiae Fructus,Paeoniae Radix Alba | 0.9956 |
|    Fraxini Cortex,Scutellariae Radix,Paeoniae Radix Alba     | 0.9954 |

## Project Support

This material is based upon work supported by PengCheng Lab.

