# kz-profanity-detection-DL26
A trained model that reads a Kazakh product review or comment about goods and predicts whether it contains Kazakh profanity

Nurlankyzy Dilnaz · DSSC5113 Deep Learning · Fall 2026

## Task

Input: one comment. Output: **0** (clean) or **1** (contains Kazakh profanity), plus a probability:
```python
predict(bundle, "Сапасы өте нашар, ақшамды қайтарыңыз")   # {'label': 0, 'probability': 0.0002}
```

## Data

The reviews come from [KazSAnDRA](https://github.com/IS2AI/KazSAnDRA) (ISSAI), a public dataset of real Kazakh reviews from app stores, marketplaces, maps and book stores ([paper](https://arxiv.org/abs/2403.19335), license: CC BY 4.0).

KazSAnDRA is labeled for sentiment, not profanity, so all its polarity files were combined into one pool of **166,577 reviews**, and profanity labels were made from scratch:

1. **Normalize** the text, so that spelling variants look the same (қ→к, ү→у, repeated letters collapsed, Latin → Cyrillic).
2. **Search** with a list of Kazakh and Russian profane roots, plus masks like `с**а`. Matches are only candidates.
3. **Label by hand:** all 2,455 candidates, 600 rude-but-clean reviews (жаман, ешек, мал…), and 300 random reviews.
4. **Add** 2,700 more random reviews as clean. Only 1 of the 300 checked random reviews had missed profanity, so these are almost certainly clean. They are used for training and validation only, never for testing.

**Result:** 6,055 comments, of which **2,457 are profane** and **3,598 are clean**.
Phone numbers, emails and usernames were replaced with tags.

Two things stood out:
- Almost all profanity is in **app-store** reviews; marketplace reviews have almost none.
- About **half of the profane reviews are positive**, so profanity is not the same as a bad review.

**Splits:** after removing duplicates, **5,979** comments remain: about 3.6% of the 166,577 review pool. Only this part was labeled, because profanity is rare (~1.5%) and hand-labeling the whole pool was not feasible. These comments were split 70 / 15 / 15 with seed 42. Each split keeps a similar share of profane comments, near-duplicates stay in the same split, and the test set contains only hand-checked rows.

| | Clean | Profane |
|---|---|---|
| Train | 2,940 | 1,664 |
| Validation | 530 | 364 |
| Test | 120 | 361 |

## Labeling rules
> Note: some vulgar words are written in full below to define the labeling boundaries precisely; strong obscenities are masked (e.g. `с**а`).

A comment is **1** if it contains an obscene or vulgar word, in any spelling.

- Negative or angry review without swear words → **0**
- Mild insults (мал, ешек, ақымақ, тупой) → **0**
- Russian swear words in a Kazakh comment → **1** (very common in Kazakh speech)
- Masked words (`с**а`) → **1** if the hidden word is clear
- Misspelled, Latin or joined-together forms → same as the normal word
- Codes and abbreviations (шшс, 667) → **1**
- боқ, говно → **1**;
- көт and its forms (көтің, көтіне, көтіңді қыс) → **1**
- көтен (child's word, common in diaper reviews) → **0**
- Talking about swearing ("боқ сөз айтпаңдар") or emoji only (🖕) → **0**

| Comment (masked) | Label |
|---|---|
| Реклама коп | 0 |
| Мал ойын тартпандар бауырлар | 0 |
| Баланың көтенін құрғақ ұстайды | 0 |
| Скотчпен қатты жабыстырып орап тастаған | 0 |
| Миллион доллар с**а | 1 |
| Жарнама жарнама Қ****бас | 1 |
| Рекламасын с*нейін шшс | 1 |

## Model

A **character-level CNN** trained from scratch in PyTorch.

Why characters: swear words are often misspelled, masked or written in Latin letters. A model that reads letter by letter still recognizes parts of them, and it is small enough to train in a few minutes without a GPU.

How it works:
1. Normalize the text, with the same function for training and prediction (`src/preprocess.py`).
2. Turn each character into a number. Rare characters become `<unk>`.
3. Cut or pad every comment to 300 characters, which covers about 98% of comments.
4. The CNN (kernel sizes 3/4/5) outputs a probability. Above the threshold → profane.

Settings: Adam, learning rate 0.001, batch 64, up to 30 epochs with early stopping (best epoch: 21), class weights for balance, seed 42. The decision threshold **0.7** was chosen on the validation set; F1 is almost the same for any threshold between 0.5 and 0.75.

## Experiments

On the validation set:

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Keyword list only | 0.997 | 0.995 | 0.996 |
| TF-IDF + logistic regression | 0.968 | 0.907 | 0.936 |
| Char-CNN without normalization | 0.961 | 0.948 | 0.954 |
| **Char-CNN (final)** | 0.991 | 0.948 | **0.969** |

- The char-CNN beats TF-IDF, mostly by catching more profanity (recall 0.95 vs 0.91).
- Normalization helps: F1 0.954 → 0.969.
- The keyword list scores highest only because the labels were found with it. Its real weakness is profanity that is not on the list.

## Results

Tested once on 481 hand-checked comments (120 clean, 361 profane):

**Precision 0.988 · Recall 0.928 · F1 0.957**

![Confusion matrix](results/confusion_matrix.png)

The model rarely raises false alarms (4 of 120 clean comments) but misses about 7% of profanity (26 of 361).

- **False positive:** a clean review is flagged and may be hidden unfairly.
- **False negative:** a swear word gets through and other users see it.

For moderation, missing profanity is usually worse. Seven of the missed comments had probabilities between 0.55 and 0.69, so a lower threshold, chosen on validation, could raise recall; borderline cases could also go to a human moderator.

New spellings work too: "kotakbas" (Latin), "қ0тақ" (digit) and "с*к*" (masked) are all detected, while "курьер ақымақ екен" (mild insult) stays clean.

## Mistakes the model made

1. "Аферист мошенниктер… кайыршылар": predicted 1, should be 0. Angry insults were mistaken for profanity.
2. "Қаншық ойын": predicted 0 (p = 0.20), should be 1. Қаншық is rare in the training data, and other rare words (жалап, жезөкше, говно) were also missed.
3. "Итин амы": predicted 0 (p = 0.69), should be 1. Just below the threshold; similar "… амы" variants were missed the same way.

Some errors are really labeling problems: "жазбапедік" ("we didn't write") was a false keyword match labeled 1, and a few others ("Қарақаншық" from a book, "шешеңді сыйлайындар", "татақ") are borderline. Test labels were not changed after evaluation.

## How to run

Open any notebook from `notebooks/` in Google Colab; the first cell clones this repository. To run locally:
```bash
git clone https://github.com/dilya17/kz-profanity-detection-DL26.git
cd kz-profanity-detection-DL26
pip install -r requirements.txt
```

1. `01_data_preparation.ipynb`: gets the data, finds candidates, builds `data/labeled.csv`
2. `02_split_and_train.ipynb`: splits, baselines, trains and saves the model
3. `03_evaluate_and_predict.ipynb`: tests once, shows errors, predicts new text

The trained model is in `model/`, so no retraining is needed:
```python
from src.predict import load_model, predict
bundle = load_model("model")
predict(bundle, "Керемет тауар, рахмет!")
```

## Limitations

- Profane examples were found with a word list, so unusual profanity is underrepresented, and rare words are often missed.
- Almost no profane marketplace reviews, so product reviews are not well covered.
- The test set is small (481) and 75% profane; precision rests on only 120 clean comments. Real reviews are about 1.5% profane, so real-world precision would be lower.
- One person labeled everything; borderline words could be judged differently by others, and a few test labels are debatable.
- The model sees letters, not meaning, so it can't tell talking about swearing from swearing itself.
- Latin text is only partly handled: letter pairs (sh, zh), official Kazakh Latin letters (ğ, ş, á, ý) and mixed-script comments are not fully converted.

Ideas for improvement: more examples of rare words, real marketplace comments, more annotators, a pretrained multilingual model, and human review of borderline cases.

## Credits

- Dataset: KazSAnDRA by ISSAI, Nazarbayev University (CC BY 4.0).
- AI assistance: AI helped with drafting the code, debugging and the README structure.
- Own work: labeling rules, extending and fixing the keyword list, checking 3,355 labels by hand, structuring the code, running the experiments and analyzing the results.
