# Phishing Email Detection using Machine Learning

A beginner-friendly end-to-end project using **Python, Scikit-learn, TF-IDF, URL/text feature engineering, Logistic Regression and Streamlit**.

## 1. Project features

- Train on labelled phishing and legitimate email text.
- Convert email content into TF-IDF features (unigrams and bigrams).
- Add URL signals (URL count, URL length, IP-based URLs, unusual subdomains and selected TLDs).
- Add language signals (urgency words, account-verification wording, message length and punctuation).
- Classify an email as **Phishing** or **Safe**.
- Report accuracy, precision, recall, F1-score and confusion matrix.
- Use a simple web interface to test pasted email text.

## 2. Project structure

```text
phishing_email_detection_project/
├── app.py
├── features.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── emails.csv              # Add your labelled dataset here
├── models/                     # Trained model is saved here
└── reports/                    # Metrics and confusion matrix are saved here
```

## 3. Dataset

Use a labelled CSV containing both phishing and legitimate emails. A public starter source is the **Phishing Validation Emails Dataset** hosted by Zenodo:

https://zenodo.org/records/13474746

Download `Phishing_validation_emails.csv`, then make a working copy named `emails.csv` inside the `data/` folder. The training script automatically recognizes common text and label column names. If the source CSV uses different column names or labels, rename/map them to:

```csv
email_text,label
"Please verify your account immediately at https://example.test",1
"Hi team, the meeting is scheduled for 10 AM tomorrow.",0
```

Label convention:
- `1` = Phishing
- `0` = Safe / legitimate

The Zenodo starter set contains 2,000 labelled messages. Check its documentation and citation/licence details before redistribution or formal research use. For a stronger project evaluation, use a larger, diverse corpus and test on a separate dataset that was not used during training. Do not report accuracy from a tiny sample as evidence of real-world performance.

## 4. Installation

Open a terminal in the project folder. Create and activate a virtual environment (recommended):

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install libraries:

```bash
pip install -r requirements.txt
```

## 5. Train the model

Put your dataset at `data/emails.csv`, then run:

```bash
python train_model.py
```

The script:
1. Loads and cleans the CSV.
2. Removes empty and duplicate email text.
3. Splits the data into stratified training and test sets (80/20).
4. Fits the TF-IDF + engineered-feature + Logistic Regression pipeline.
5. Prints evaluation metrics.
6. Saves the trained model to `models/phishing_model.joblib`.
7. Saves metrics and confusion matrix to `reports/`.

To use a different CSV path:

```bash
python train_model.py --data path/to/your_dataset.csv
```

## 6. Run the web application

After training:

```bash
streamlit run app.py
```

Streamlit opens the detector in your browser. Paste the email text and click **Analyze Email**.

## 7. Understanding the evaluation

- **Accuracy:** proportion of all test emails classified correctly.
- **Precision:** among emails predicted as phishing, how many were actually phishing.
- **Recall:** among actual phishing emails, how many the model detected.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions for each class.

For phishing detection, recall is especially important because a false negative means a phishing email was classified as safe. Precision also matters because too many false alarms make a detector difficult to use.

## 8. Important limitations

This is an educational baseline, not a production email-security product. URL features are heuristic; the model does not open URLs, check domain reputation, authenticate senders, inspect attachments or parse full email headers. A high test score does not guarantee safety. Dataset duplicates, source overlap, synthetic examples and class imbalance can inflate metrics. For a serious evaluation, use deduplication and source-aware/time-aware test splits, and test on new real-world messages.

## 9. Suggested future enhancements

- Compare Logistic Regression, Multinomial Naive Bayes, Random Forest and Linear SVM.
- Add sender-domain and email-header features.
- Add URL reputation checks using a safe, authorized threat-intelligence API.
- Add explainability showing the words/features influencing a prediction.
- Evaluate on a held-out external dataset and report precision-recall curves.

## References

- Zenodo, *Phishing Validation Emails Dataset*: https://zenodo.org/records/13474746
- Scikit-learn text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit documentation: https://docs.streamlit.io/
