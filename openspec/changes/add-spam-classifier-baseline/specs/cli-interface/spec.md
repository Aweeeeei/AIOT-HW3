## ADDED Requirements

### Requirement: Training Script Interface
The system SHALL provide a training script with clear CLI options.

#### Scenario: Basic Training
- **WHEN** running train.py
- **THEN** it requires --input and --output paths
- **AND** accepts optional --seed parameter
- **AND** displays progress information

### Requirement: Prediction Script Interface
The system SHALL provide a prediction script for inference.

#### Scenario: Message Classification
- **WHEN** running predict.py
- **THEN** it requires --model and input text/file
- **AND** supports single or batch prediction
- **AND** formats output clearly