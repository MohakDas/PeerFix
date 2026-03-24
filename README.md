# PeerFix

PeerFix is an NLP-based peer-review triage assistant that helps authors analyze reviewer comments by predicting whether a comment is actionable, identifying its comment type, and mapping it to a practical response strategy.

Built using the ReAct peer-review dataset, PeerFix compares classical ML baselines with fine-tuned transformer models and provides a simple Streamlit dashboard for interactive use.

# Features
- Predicts whether a reviewer comment is actionable or non-actionable
- Classifies actionable comments into types such as:
- shortcoming
- suggestion
- question
- fact
- agreement
- disagreement
## Maps predictions into a triage layer:
- priority level
- response mode
- short guidance
_Includes a Streamlit interface for single-comment analysis_
## Compares:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Fine-tuned RoBERTa

# Motivation

Handling peer-review feedback can be time-consuming and mentally noisy, especially when authors receive multiple comments with different levels of urgency and usefulness. PeerFix is designed to help prioritize reviewer feedback and support faster, more structured decision-making during paper revision.

# Dataset

This project uses the **ReAct dataset**, a curated corpus of reviewer comments from ICLR peer reviews annotated for:

- Label1: actionable / non_actionable
- Label2: agreement, disagreement, suggestion, question, shortcoming, fact, other

Dataset source: [ReAct Dataset](https://github.com/gtmdotme/ReAct)

# Models Used
- Classical baselines
- Logistic Regression for actionability classification
- Linear SVM for comment-type classification

## Transformer models
- RoBERTa fine-tuned for actionability classification
- RoBERTa fine-tuned for comment-type classification

# Results
- Actionability classification
- Logistic Regression baseline: 0.68 accuracy
- RoBERTa fine-tuned: 0.84 accuracy, 0.84 weighted F1
## Comment type classification
- Logistic Regression baseline: 0.54 accuracy
- Linear SVM baseline: 0.59 accuracy
- RoBERTa fine-tuned: 0.916 accuracy, 0.908 weighted F1

# How It Works

PeerFix uses a two-stage pipeline:

Actionability classifier predicts whether the comment needs action.
If the comment is actionable, the comment-type classifier predicts the type of reviewer comment.
The output is passed through a triage rule layer to produce:
- priority
- response mode
- guidance

_This turns raw reviewer comments into more usable author-side decision support._

# Notes
- Trained model checkpoints are not included in this repository due to size.
- To reproduce the results, run the training scripts inside the training/ folder.
- The app expects trained models to be saved under:
    - training/actionability_model/final
    - training/comment_type_model/final

# Future Improvements
- Batch analysis for multiple reviewer comments
- Confidence-aware warnings for uncertain predictions
- Invalid or out-of-domain comment detection
- Exportable triage summaries
- Better handling of highly imbalanced comment classes

# Author

Mohak Das

GitHub: [MohakDas](https://github.com/MohakDas)

# License

This project uses the ReAct dataset. Please follow the dataset’s original license and citation requirements before reuse in research or redistribution.

 [![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
