# Code-Mixed-Machine-Translation
# CoMeT project NLP

### Dataset used: 
https://drive.google.com/file/d/1yfe4ML4JOcjC2M0AFgrSaJYqXLoCv7F_/view?usp=sharing

## Training

> Make sure to run the below commands inside training directory.

Install the dependencies using

```bash
conda env create --file environment.yml
conda activate cmtranslation2
```

Download mBART pre-trained checkpoint:

```bash
wget -c https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz

```
### Commands to run:

#### data_code.py:
```bash
python3 data_code.py <MODEL>
```
MODELs can be either mBARTen or mBARThien

### Vocabulary_build.py:
```bash
python3 Vocabulary_build.py --corpus-data "./ft/*.spm.*" --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/dict.txt
```

### fine_tune.py:
```bash
python3 fine_tune.py --pre-train-dir ./mbart.cc25.v2 --ft-dict ./ft/dict.txt --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --output ./ft/model.pt
```
### evals
Install the dependencies for evaluation(the same as training) and run evaluation:

```bash
conda env create --file eval-environment.yml
conda activate cmtranslation2
bash eval.sh <temporary directory> <path to dataset directory> // Replace with below cmd
```
```bash
bash eval.sh ../training/data/preprocessed ../training/data/
```
DATA_DIR: training/data/
SCRATCH_DIR: training/data

```bash
conda env create --file scoring-environment.yml
conda activate indictrans
python3 calc_scores.py <temporary directory> <path to dataset directory> // Replace with below cmd
```

python3 calc_scores.py ../training/data ../training/data/
