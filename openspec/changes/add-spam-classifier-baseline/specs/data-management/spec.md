## ADDED Requirements

### Requirement: CSV Data Loading
The system SHALL load and parse CSV files containing spam/ham messages.

#### Scenario: File Reading
- **WHEN** loading a CSV file
- **THEN** it handles headerless format
- **AND** identifies message and label columns
- **AND** reports the number of loaded samples

### Requirement: Basic Preprocessing
The system SHALL apply consistent text preprocessing.

#### Scenario: Text Normalization
- **WHEN** preprocessing messages
- **THEN** it converts to lowercase
- **AND** removes excess whitespace
- **AND** handles basic special characters