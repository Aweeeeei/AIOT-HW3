# Text Classification Capability

## Purpose
Provide a machine learning pipeline for classifying text messages as spam or ham (not spam) with high accuracy and reproducible results.

### Requirement: Model Training
The system SHALL provide a training pipeline that processes input data and produces a reusable classification model.

#### Scenario: Basic Training Flow
- **WHEN** the training script is run with a CSV dataset
- **THEN** it creates a TF-IDF vectorizer and logistic regression model
- **AND** saves both artifacts using joblib
- **AND** reports training metrics (Accuracy, Precision, Recall, F1)

#### Scenario: Reproducible Training
- **WHEN** the training script is run multiple times with the same seed
- **THEN** it produces identical model artifacts
- **AND** achieves the same performance metrics

### Requirement: Text Classification
The system SHALL classify new text messages as spam or ham using the trained model.

#### Scenario: Single Message Classification
- **WHEN** a new message is provided to the prediction script
- **THEN** it loads the trained model
- **AND** returns a binary classification (spam/ham)
- **AND** provides a confidence score

#### Scenario: Batch Classification
- **WHEN** multiple messages are provided in a CSV file
- **THEN** the system classifies all messages
- **AND** returns results in a structured format
- **AND** maintains performance under 2 minutes for processing

### Requirement: Performance Metrics
The system SHALL provide clear performance metrics for model evaluation.

#### Scenario: Metric Reporting
- **WHEN** the model completes training
- **THEN** it reports accuracy on the test set
- **AND** provides precision and recall for spam detection
- **AND** calculates the F1 score

#### Scenario: Hold-out Validation
- **WHEN** training the model
- **THEN** it uses an 80/20 train-test split
- **AND** evaluates performance on the held-out test set