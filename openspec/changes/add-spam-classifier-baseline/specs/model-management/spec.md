## ADDED Requirements

### Requirement: Basic Model Storage
The system SHALL save trained models using joblib.

#### Scenario: Model Saving
- **WHEN** training completes successfully
- **THEN** it saves vectorizer and classifier
- **AND** uses .joblib extension
- **AND** includes timestamp in filename

### Requirement: Model Loading
The system SHALL load saved models for prediction.

#### Scenario: Load for Inference
- **WHEN** loading a saved model
- **THEN** it restores vectorizer and classifier
- **AND** verifies file exists
- **AND** handles loading errors gracefully