# Corso di apprendimento PyTorch

[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)

---

## Informazioni

Questo repository è un progetto pratico di apprendimento per PyTorch, che esplora i passaggi chiave nel workflow del deep learning:

- Caricamento e gestione dei dataset
- Creazione di DataLoader
- Definizione e addestramento di reti neurali
- Salvataggio dei modelli ed esecuzione di inferenza

Tutto in questo progetto è ispirato e segue i tutorial e guide ufficiali di PyTorch, adattato per un apprendimento passo passo.

> ⚠️ Nota: il progetto è in aggiornamento continuo. Nuovi tutorial, notebook e script possono essere aggiunti nel tempo, quindi i contenuti e i nomi delle cartelle potrebbero variare.

---

##  Struttura del Progetto

Struttura attuale (soggetta a modifiche):

- `[00]_dataset_and_dataloader/` – Esplorazione di Dataset e DataLoader
- `[01]_model_creation_and_train/` – Creazione e addestramento di reti neurali
- `[02]_model_loading_and_inference/` – Caricamento di modelli addestrati e inferenza
- `[03]_first_exercise/` - Esercizio che combina i tre tutorial precedenti

> Ogni cartella contiene un **Jupyter Notebook** e uno script `main.py`.
> Dataset e modelli salvati sono ignorati da git (`data/` e `model/`) per mantenere leggero il repository.

---

## Installazione

Setup consigliato con **conda**:

### Creare un nuovo ambiente
```bash
conda create -n pytorch_course python=3.10
conda activate pytorch_course
```

### Installare PyTorch con CUDA (se disponibile)
Segui la guida ufficiale qui: https://pytorch.org/get-started

### Dipendenze aggiuntive
```bash
pip install -r requirements.txt
```
---

### Come Usarlo

1. Aprire i notebook Jupyter
2. Seguire i notebook passo passo
3. In alternativa, eseguire direttamente gli script Python:
```bash
python <script_name>.py
```

> Assicurati che la cartella `data/` sia presente o lascia che gli script scarichino automaticamente i dataset.

---

### Punti Chiave

- API di PyTorch Dataset e DataLoader
- Creazione di **dataset personalizzati**
- Costruzione e addestramento di **reti neurali feedforward**
- Uso di **funzioni di perdita** e **ottimizzatori** (`SGD` / `Adam`)
- Valutazione dei modelli e **visualizzazione delle predizioni**
- Salvataggio e caricamento dei pesi dei modelli per inferenza
- Costruzione di **reti neurali convoluzionali**

---

## Riferimenti

Tutorial PyTorch: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

---

## Contribuire

Questo progetto è pensato per apprendimento e sperimentazione.  
Sentiti libero di fare fork, sperimentare e migliorarlo!  
Poiché il progetto è in aggiornamento continuo, torna a controllare per nuovi tutorial ed esempi.

---