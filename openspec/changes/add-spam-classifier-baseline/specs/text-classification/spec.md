## ADDED Requirements

### Requirement: TF-IDF Vectorization
The system SHALL convert text messages into TF-IDF feature vectors for classification.

#### Scenario: Feature Extraction
- **WHEN** processing text messages
- **THEN** it creates TF-IDF vectors
- **AND** handles new vocabulary at prediction time
- **AND** maintains consistent preprocessing

### Requirement: Logistic Regression Training
The system SHALL train a logistic regression model on TF-IDF features.

#### Scenario: Model Training
- **WHEN** training with labeled data
- **THEN** it fits a logistic regression classifier
- **AND** uses default scikit-learn parameters
- **AND** completes training in under 2 minutes

### Requirement: Basic Metrics
The system SHALL calculate and report standard classification metrics.

#### Scenario: Performance Reporting
- **WHEN** training completes
- **THEN** it prints accuracy, precision, recall, F1
- **AND** includes the number of training samples
- **AND** shows class distribution statistics