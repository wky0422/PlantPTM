# `<div align="center"><img src="images/logo.png" width="80"></div>`

>This is a repository containing source code for paper titled "PlantPTM: Accurate prediction of diverse PTM sites in plant with multi-view deep learning and protein language models".

## Introduction

Post-translational modifications (PTMs) function as molecular switches that play crucial roles in regulating plant growth and development, stress adaptation, and controlling protein degradation. However, mass spectrometry-based techniques remain time-consuming and labor-intensive, and a reliable and cost-effective computational method for predicting plant PTM sites remains lacking. Therefore, to advance PTM studies in plants, we present PlantPTM, an integrated deep learning framework for predicting nine types of plant PTMs, including N-Glycosylation (Ngly), S-Acylation (Sacy), 2-Hydroxyisobutyrylation (Khib), Crotonylation (Kcr), Succinylation (Ksucc), Malonylation (Kmal), Acetylation (Kac), Ubiquitination (Kub), and Phosphorylation (pho). To our knowledge, PlantPTM represents the first computational tool specifically designed for broad, multi-type PTM site prediction across plants. Through independent testing against existing general or plant-specific prediction tools, extensive comparative experiments demonstrated that PlantPTM outperforms existing PTM tools by an average of 42.89%, with 2.90% to 19.08% improvement over the best-performing tools. Moreover, PlantPTM demonstrates robust performance across datasets of varying scales. Notably, PlantPTM can be integrated with AlphaFold3. We used PlantPTM to predict all potential pho sites in the Gα subunit (GPA1) of *Arabidopsis* G proteins, enabling further prediction of G protein structures based on known pho sites. Among the nine PTM types currently supported by PlantPTM, five (Ngly, Sacy, Kcr, Kac, and pho) can be further processed through AlphaFold3 to generate PTM-modified protein structures. The web-server of PlantPTM is freely accessible at [PlantPTM server](https://ai4bio.online/PlantPTM/home/).

## PlantPTM method

### Model architecture

In this work, we propose PlantPTM, a deep learning framework that integrates protein language models with a diverse set of sequence-derived features. The architecture of PlantPTM consists of three core components: 

(i) a multi-encoder module incorporating integration

The multi-encoder module integrates four parallel encoding branches, each engineered to extract complementary feature representations from protein sequences;

(ii) a feature fusion module

The Feature Fusion Module is designed to intelligently integrate multi-modal features derived from all encoder outputs;

(iii) a prediction decoder module

The Prediction Decoder Module operates on the fused feature representation through a sequential three-layer fully connected architecture.

![PlantPTM overall framework](./images/PlantPTM.png)

### Requirements

Python version: >=`3.8`, < `3.12` (recommended: `3.10`)

Several common packages are used in this project.

1. For data processing: python built-in packages such as `re` and `Biopython`;
2. For deep learning model implementation: `PyTorch 2.4.1 + cu121` for deep neural networks, `transformers 4.19.2` for pre-trained models;
3. For model evaluations and result plots:`scikit-learn` and `matplotlib`.

Using the following commands to create your own conda environment:

First make sure you have [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) or [miniconda](https://docs.anaconda.com/free/miniconda/) installed in your machine  using command `which conda` (for Linux).


#### Step1 Preparation for blast and ProtBert

1. Please go to blast ( [link](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) | [guide](https://ftp.ncbi.nlm.nih.gov/pub/factsheets/HowTo_BLASTGuide.pdf) ) for local deployment;
2. Please go to ProtBert ( [link](https://huggingface.co/Rostlab/prot_bert) | [guide](https://github.com/agemagician/ProtTrans) ) for local deployment.


#### Step2 Construct the uniref50 database and generate PSSM features

1. Please go to [UniPort](https://www.uniprot.org/) to download the [uniref50 database](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/): `uniref50.fasta.gz` 
2. Local construction of uniref50 database:
   ```sh
   gunzip uniref50.fasta.gz
   makeblastdb -in uniref50.fasta -dbtype prot -parse_seqids -out uniref50
   ```
3. You can use the [provided PSSM features](https://github.com/wky0422/PlantPTM/tree/main/data/pssm) or regenerate using the following methods：
   ```sh
   psiblast -query input.fasta -db uniref50 -num_iterations 3 -out_ascii_pssm output.pssm
   ```

#### Step3 Conda environment required for PlantPTM

1. The environment required for deploying PlantPTM：
   ```sh
   conda create -n PlantPTM python==3.10 && conda activate SLAM
   pip install numpy pandas biopython transformers scikit-learn matplotlib seaborn tqdm
   ```
2. You can install via Anaconda:

   ```sh
   conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install torch_geometric
   conda install pyg=*=*cu* -c pyg
   ```

   Or install by wheels:

   ```sh
   pip install torch==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
   pip install torch_geometric
   pip install torch_scatter torch_cluster --no-index -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   ```

   If you are running on a CPU-only machine, it is recommended to install the CPU version of these packages using following commands:

   ```sh
   pip install torch==2.4.1
   pip install torch_geometric
   pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
   ```

   Finally, check the installations by:

   ```python
   python -c "import torch; print(torch.cuda.is_available())"	# Should be True
   python -c "import torch; print(torch.cuda.device_count())"	# Should be > 0
   python -c "import torch; print(torch.__version__);print(torch.version.cuda)"
   ```

   For more information, please refer to the official documentation of [PyTorch ](https://pytorch.org/get-started/locally/)and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

☆   **Alternative**
   
>You can simply  can use this [`requirements.txt`](https://github.com/wky0422/PlantPTM/blob/main/requirements.txt) file to install all the dependencies.

>```sh
>pip install -r requirements.txt
>```
   
>You can read more in [Virtualenv and pip Basics](http://jonathanchu.is/posts/virtualenv-and-pip-basics/), and the official [Requirements File Format](https://pip.pypa.io/en/stable/cli/pip_install/) documentation.

## Descriptions of this repository

1. codes
   - [Dataset.py](https://github.com/wky0422/PlantPTM/blob/main/codes/Dataset.py): dataset preparation and processing
   - [Generate_PLM_embeddings.py](https://github.com/wky0422/PlantPTM/blob/main/codes/Generate_PLM_embeddings.py): generation of full-length protein features
   - [Metrics.py](https://github.com/wky0422/PlantPTM/blob/main/codes/Metrics.py): validation metrics and additional utils
   - [Model.py](https://github.com/wky0422/PlantPTM/blob/main/codes/Model.py): PyTorch-version PlantPTM model
   - [PlantPTM.py](https://github.com/wky0422/PlantPTM/blob/main/codes/PlantPTM.py): for PlantPTM training
   - [Predict.py](https://github.com/wky0422/PlantPTM/blob/main/codes/Predict.py): for predicting PTMs using PlantPTM
   - [Threshold.txt](https://github.com/wky0422/PlantPTM/blob/main/codes/Threshold.txt): for predicting using PlantPTM with different model thresholds

3. data
   - [fasta](https://github.com/wky0422/PlantPTM/tree/main/data/fasta): PTMs dataset clustered with 30% identity
   - [pssm](https://github.com/wky0422/PlantPTM/tree/main/data/pssm): compressed file for storing PSSM feature files corresponding to each protein

4. model
   - Please download the pickled models from [PlantPTM server](https://ai4bio.online/PlantPTM/download/).
    
5. case study on GPA1

   - [case study](https://github.com/wky0422/PlantPTM/tree/main/case_study): We predicted all potential pho sites of G α (GPA1) in Arabidopsis G protein using PlantPTM, and further utilized AlphaFold3 to perform structural predictions for Gα-Gβ-Gγ based on experimentally-verified phosphorylation sites in the Gα subunit, displaying the C-terminal domain of all potential phosphorylation sites.

![GPA1](./images/GPA1.png)

## Citation

If you find PlantPTM helpful in your study, please kindly cite the following [paper ]():

*[APA]*

Dou, G., Wang, K., … Chen, Z. (2025). PlantPTM: Accurate prediction of diverse PTM sites in plant with multi-view deep learning and protein language models. *bioRxiv.org*, ×, ×. doi:×××

or

BibTeX entry:

```bibtex
@article{Wang,
title = {PlantPTM: Accurate prediction of diverse PTM sites in plant with multi-view deep learning and protein language models},
journal = {*bioRxiv.org*},
volume = {×},
pages = {×},
year = {×},
issn = {×},
doi = {×××},
url = {×××},
author = {Gensheng Dou and Kaiyuan Wang and ... and Zhen Chen},
keywords = {},
abstract = {}
}
```

## License

This repository is licensed under the terms of the **Apache 2.0** [license](https://github.com/wky0422/PlantPTM/blob/main/LICENSE).
