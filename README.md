# Azure AI Content Understanding Demo

This demo package provides comprehensive examples and utilities for working with Azure AI Content Understanding service. It demonstrates document analysis, audio processing, and custom analyzer creation capabilities.

## Quick Start

### Prerequisites

1. **Azure AI Foundry Resource**: Create an Azure AI Foundry resource in the Azure portal
2. **Python 3.11+**: Ensure you have Python 3.11 or later installed
3. **Dependencies**: Install required packages using pip

### Setup

1. **Clone or download this demo package**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
   - Copy `.env.template` to `.env`
   - Fill in your Azure AI Service endpoint and credentials:
   ```
   AZURE_AI_SERVICE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_AI_SERVICE_API_KEY=your-api-key-here
   ```

4. **Run the interactive notebook**:
```bash
jupyter notebook content_understanding_demo_notebook.ipynb
```

Or run the comprehensive demo:
```bash
python scenarios/comprehensive_demo.py
```

## Package Structure

```
demo/
├── requirements.txt              # Python dependencies
├── .env.template                # Environment configuration template
├── content_understanding_client.py   # Core client library
├── content_understanding_demo_notebook.ipynb # Interactive Jupyter demo
├── create_audio.ipynb           # Audio creation notebook
├── scenarios/                   # Demo scenarios
│   ├── document_analysis_demo.py     # Document analysis examples
│   ├── audio_analysis_demo.py        # Audio analysis examples
│   ├── healthcare_assessment_demo.py # Healthcare assessment automation
│   ├── golf_coaching_assessment_demo.py # Golf coaching streaming analysis
│   ├── comprehensive_demo.py         # Main demo script
│   ├── batch_processing_demo.py      # Batch processing examples
│   └── rag_integration_demo.py       # RAG integration examples
├── utils/                       # Utility modules
│   ├── healthcare_utils.py          # Healthcare compliance utilities
│   ├── golf_coaching_utils.py       # Golf analytics and streaming utilities
│   ├── golf_analyzer.py             # Golf performance analysis
│   ├── golf_models.py               # Golf assessment data models
│   ├── assessment_processing.py     # Assessment processing utilities
│   └── utils.py                     # General utility functions
├── evaluate/                    # Evaluation framework
│   ├── evaluation.py                # Core evaluation engine with semantic similarity
│   ├── evaluate_assessments.py      # Batch evaluation runner
│   └── example_evaluation.py        # Usage examples
├── audio/                       # Sample audio files
├── doc/                         # Sample documents
├── groundtruth/                 # Ground truth data for evaluation
├── prompts/                     # System prompts and templates
├── output/                      # Generated results and reports
├── run_demo.bat                 # Windows demo runner
├── run_demo.sh                  # Unix demo runner
└── README.md                    # This file
```

## Demo Scripts

### 0. Interactive Jupyter Notebook (`content_understanding_demo_notebook.ipynb`)
The main interactive demo that showcases all capabilities in a notebook format:
- Step-by-step guided experience
- Document and audio analysis examples
- Healthcare and golf assessment scenarios
- Built-in evaluation framework demonstration
- Real-time results visualization

**Run it:**
```bash
jupyter notebook content_understanding_demo_notebook.ipynb
```

### 1. Comprehensive Demo (`scenarios/comprehensive_demo.py`)
The main demo script that showcases all capabilities:
- Configuration validation
- Document analysis with prebuilt analyzers
- Custom analyzer creation and usage
- Audio processing and transcription
- Analyzer management operations

**Run it:**
```bash
python scenarios/comprehensive_demo.py
```

### 2. Document Analysis Demo (`scenarios/document_analysis_demo.py`)
Focused on document processing capabilities:
- PDF, image, and Office document analysis
- Custom field extraction
- Document structure analysis

**Run it:**
```bash
python scenarios/document_analysis_demo.py
```

### 3. Audio Analysis Demo (`scenarios/audio_analysis_demo.py`)
Demonstrates audio processing features:
- Audio transcription and diarization
- Call center analysis
- Meeting audio processing
- Custom audio analyzer creation

**Run it:**
```bash
python scenarios/audio_analysis_demo.py
```

### 4. Batch Processing Demo (`scenarios/batch_processing_demo.py`)
Shows how to process multiple files efficiently:
- Concurrent processing
- Progress tracking
- Error handling and retry logic

**Run it:**
```bash
python scenarios/batch_processing_demo.py
```

### 5. RAG Integration Demo (`scenarios/rag_integration_demo.py`)
Demonstrates integration with RAG (Retrieval-Augmented Generation) systems:
- Content extraction for knowledge bases
- Azure AI Search integration
- Vector embedding preparation

**Run it:**
```bash
python scenarios/rag_integration_demo.py
```

### 6. Healthcare Assessment Demo (`scenarios/healthcare_assessment_demo.py`)
Shows automated patient assessment form completion from care staff-patient conversations:
- Patient assessment automation
- Clinical data extraction
- HIPAA-compliant processing
- Multiple assessment types (mental health, physical health, geriatric)

**Run it:**
```bash
python scenarios/healthcare_assessment_demo.py
```

### 7. Golf Coaching Assessment Demo (`scenarios/golf_coaching_assessment_demo.py`)
Demonstrates real-time streaming audio analysis for golf coaching conversations:
- Real-time question-answer detection from streaming audio
- Automatic assessment form completion
- Golf-specific performance analysis
- Multiple skill level scenarios (beginner, intermediate, advanced)
- Coaching recommendations and practice plans

**Run it:**
```bash
python scenarios/golf_coaching_assessment_demo.py
```

### 8. Assessment Evaluation Framework (`evaluate/`)
Comprehensive evaluation system for validating assessment extraction quality:
- Semantic similarity using TF-IDF and cosine similarity (scikit-learn)
- Multiple evaluation metrics (exact match, token F1, BLEU, field coverage)
- Domain-specific evaluation for healthcare and golf assessments
- Support for ignoring missing fields in ground truth comparisons
- Batch evaluation capabilities with detailed reporting

**Run healthcare evaluation:**
```bash
python evaluate/evaluate_assessments.py --demo healthcare --ground-truth mental.json --ignore-missing-fields
```

**Run golf evaluation:**
```bash
python evaluate/evaluate_assessments.py --demo golf --ground-truth golf.json --ignore-missing-fields
```

**Run all evaluations:**
```bash
python evaluate/evaluate_assessments.py --all --ignore-missing-fields
```

**Features:**
- Real-time streaming audio processing
- Automatic question-answer detection from conversations
- Golf-specific assessment forms and field mapping
- Performance analysis and skill level assessment
- Coaching recommendations and structured practice plans
- Multiple player scenarios (beginner to advanced)

- Clinical alerts and recommendations
- HIPAA compliance utilities

## Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
# Required
AZURE_AI_SERVICE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_AI_SERVICE_API_KEY=your-api-key-here

# Optional
AZURE_AI_SERVICE_API_VERSION=2025-05-01-preview
DEFAULT_OUTPUT_DIR=./output
LOG_LEVEL=INFO
```

### Authentication Options

1. **API Key Authentication** (Recommended for demos):
   ```bash
   AZURE_AI_SERVICE_API_KEY=your-api-key
   ```

2. **Azure AD Authentication** (Recommended for production):
   ```bash
   # Set these for service principal authentication
   AZURE_TENANT_ID=your-tenant-id
   AZURE_CLIENT_ID=your-client-id
   AZURE_CLIENT_SECRET=your-client-secret
   ```

## Sample Data

The demos use publicly available sample files:

- **Documents**: PDF invoices, receipts, contracts
- **Audio**: Call recordings, meeting audio
- **Images**: Charts, diagrams, scanned documents

You can replace these with your own files by updating the URLs in the demo scripts.

## Features Demonstrated

### Document Analysis
- ✅ Text extraction from PDFs and images
- ✅ Table and structure detection
- ✅ Custom field extraction (invoices, contracts)
- ✅ Handwriting recognition
- ✅ Mathematical formula detection

### Audio Analysis
- ✅ Speech-to-text transcription
- ✅ Speaker diarization
- ✅ Multilingual support
- ✅ Call center analytics
- ✅ Meeting summarization

### Custom Analyzers
- ✅ Field schema definition
- ✅ Business-specific extraction
- ✅ Classification and categorization
- ✅ Confidence scoring

### Integration Capabilities
- ✅ Batch processing
- ✅ RAG system integration
- ✅ Azure AI Search connectivity
- ✅ Error handling and retry logic

## Healthcare Use Case

The healthcare assessment demo demonstrates automated patient assessment form completion from care staff-patient conversations. This showcases real-world application of Content Understanding in clinical settings.

### Features

- **Multiple Assessment Types**: Mental health screening, physical health assessment, geriatric assessment
- **Clinical Data Extraction**: Automatically extracts relevant clinical information from conversations
- **HIPAA Compliance**: Built-in utilities for data anonymization and secure handling
- **Confidence Scoring**: Validates extraction confidence for clinical decision support
- **Clinical Alerts**: Flags high-risk conditions requiring immediate attention
- **EHR Integration**: Export capabilities to standard healthcare formats (HL7 FHIR)
- **Quality Evaluation**: Semantic similarity evaluation for clinical term matching

### Assessment Templates

1. **Mental Health Screening**
   - Mood and anxiety assessment
   - Sleep quality evaluation
   - Risk assessment and intervention planning

2. **Physical Health Assessment**
   - Chief complaint identification
   - Pain level and mobility status
   - Functional capacity evaluation

3. **Geriatric Assessment**
   - Cognitive status evaluation
   - Fall risk assessment
   - Living situation and caregiver support

### Healthcare Utilities (`utils/healthcare_utils.py`)

- **HIPAAComplianceHelper**: Data anonymization and audit logging
- **ClinicalDataValidator**: Assessment validation and consistency checking
- **Assessment Templates**: Expandable library for different specialties
- **EHR Export**: Standard healthcare format conversion

### Usage Example

```python
from scenarios.healthcare_assessment_demo import PatientAssessmentAnalyzer

# Initialize analyzer
analyzer = PatientAssessmentAnalyzer()

# Process patient conversation
result = analyzer.analyze_patient_conversation(
    audio_file_path="patient_conversation.wav",
    assessment_type="mental_health_screening",
    patient_id="PATIENT_001",
    care_staff_id="STAFF_001"
)

# Validate results
validation = analyzer.validate_assessment(result)
print(f"Assessment valid: {validation['valid']}")
```

## Golf Coaching Use Case

The golf coaching assessment demo showcases real-time streaming audio analysis for golf coaching conversations, automatically detecting questions and answers to fill out comprehensive golf performance assessment forms.

### Features

- **Real-time Stream Processing**: Processes streaming audio from coaching conversations in real-time
- **Automatic Q&A Detection**: Intelligently identifies questions from coaches and answers from athletes
- **Smart Field Mapping**: Maps detected conversations to specific assessment form fields
- **Performance Analysis**: Analyzes golf performance data and generates coaching insights
- **Multiple Skill Levels**: Supports assessment scenarios for beginner, intermediate, and advanced players
- **Coaching Recommendations**: Generates structured practice plans and improvement strategies

### Golf Assessment Categories

1. **Player Background & Experience**
   - Years playing golf
   - Current handicap and skill level
   - Playing frequency and home course
   - Previous instruction experience

2. **Technical Skills Assessment**
   - Strongest and weakest clubs/areas
   - Driving accuracy and distance
   - Short game confidence
   - Putting consistency

3. **Mental Game & Course Management**
   - Pre-shot routine consistency
   - Pressure handling strategies
   - Course strategy preferences
   - Mental toughness and resilience

4. **Physical Fitness & Conditioning**
   - Flexibility and mobility
   - Fitness routine and stamina
   - Injuries or physical limitations
   - Endurance during play

5. **Goals & Motivation**
   - Primary improvement objectives
   - Target handicap goals
   - Practice time availability
   - Competitive interests

### Golf Coaching Utilities (`utils/golf_coaching_utils.py`)

- **StreamingAudioAnalyzer**: Real-time audio processing and conversation flow analysis
- **GolfPerformanceAnalyzer**: Performance insights and skill level assessment
- **PlayerProfile**: Comprehensive golfer profile management
- **Practice Plan Generator**: Structured practice recommendations based on assessment

### Usage Example

```python
from scenarios.golf_coaching_assessment_demo import GolfCoachingAnalyzer

# Initialize analyzer
analyzer = GolfCoachingAnalyzer()

# Analyze golf conversation
result = analyzer.analyze_golf_conversation(
    audio_url="https://example.com/golf_conversation.wav",
    assessment_type="swing_analysis"
)

# Generate assessment report
report = analyzer._display_assessment_summary(result)
print(report)
```

## Assessment Evaluation Framework

The evaluation framework provides comprehensive quality assessment for extracted assessments using multiple metrics and semantic similarity.

### Key Features

- **Semantic Similarity**: Uses TF-IDF vectorization and cosine similarity (scikit-learn) for lightweight semantic matching
- **Multiple Metrics**: Exact match, token F1, BLEU score, field coverage, and domain-specific metrics
- **Missing Field Handling**: Option to ignore missing fields in ground truth comparisons
- **Domain-Specific Evaluation**: Specialized metrics for healthcare and golf assessments
- **Batch Processing**: Evaluate multiple assessments with detailed reporting
- **Confidence Scoring**: Overall confidence assessment for extraction quality

### Evaluation Metrics

1. **Exact Match**: Binary accuracy for field-level exact matches
2. **Semantic Similarity**: TF-IDF + cosine similarity for meaning-based comparison
3. **Token F1**: Token-level precision and recall scoring
4. **BLEU Score**: Text similarity metric for natural language fields
5. **Field Coverage**: Percentage of required fields successfully extracted
6. **Healthcare-Specific**: Clinical term matching with emphasis on semantic similarity
7. **Golf-Specific**: Technical golf terminology and key field accuracy

### Usage Examples

**Evaluate Healthcare Assessment:**
```bash
python evaluate/evaluate_assessments.py --demo healthcare --ground-truth mental.json --ignore-missing-fields
```

**Evaluate Golf Assessment:**
```bash
python evaluate/evaluate_assessments.py --demo golf --ground-truth golf.json --ignore-missing-fields
```

**Programmatic Usage:**
```python
from evaluate.evaluation import AssessmentEvaluator

evaluator = AssessmentEvaluator()
result = evaluator.evaluate_assessment(
    predicted=extracted_data,
    ground_truth=reference_data,
    assessment_type="healthcare",
    metrics=["exact_match", "semantic_similarity", "custom_healthcare"],
    ignore_missing_fields=True
)

print(f"Overall Score: {result.overall_score:.3f}")
print(f"Semantic Similarity: {result.metric_results[1].score:.3f}")
```

## Output

Demo results are saved to the `./output` directory:

```
output/
├── demo_report.json              # Overall demo summary
├── prebuilt_analysis_result.json # Prebuilt analyzer results
├── custom_analysis_result.json   # Custom analyzer results
├── audio_analysis_result.json    # Audio processing results
├── evaluations/                  # Assessment evaluation reports
│   ├── healthcare_mental_evaluation.json
│   └── golf_golf_evaluation.json
├── patient_assessments/          # Healthcare assessment results
│   ├── mental_health_screening_001.json
│   └── report_mental_health_screening_001.txt
├── golf_assessments/             # Golf coaching assessment results
│   └── golf_swing_analysis_001.json
├── batch_results/               # Batch processing outputs
└── audit_log.json              # HIPAA compliance audit log
```

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your API key in `.env`
   - Check endpoint URL format
   - Ensure resource is active in Azure portal

2. **File Processing Errors**
   - Check file size limits (see service-limits.md)
   - Verify file format is supported
   - Ensure URLs are publicly accessible

3. **Timeout Issues**
   - Increase timeout values for large files
   - Check network connectivity
   - Consider using batch processing for multiple files

### Error Messages

- `HTTP 401`: Check authentication credentials
- `HTTP 429`: Rate limiting - reduce concurrent requests
- `HTTP 413`: File too large - check size limits
- `HTTP 415`: Unsupported file format

### Performance Tips

1. **For Large Files**:
   - Use smaller files when possible (<300MB for audio)
   - Consider file compression
   - Use batch processing for multiple files

2. **For Better Accuracy**:
   - Use high-quality source files
   - Specify correct language locales
   - Use appropriate analyzer for content type

## Additional Resources

- [Azure AI Content Understanding Documentation](../README.md)
- [REST API Reference](https://docs.microsoft.com/en-us/rest/api/contentunderstanding/)
- [Service Limits and Quotas](../service-limits.md)
- [Language and Region Support](../language-region-support.md)

## Next Steps

1. **Start with the Interactive Notebook**: Open `content_understanding_demo_notebook.ipynb` for a guided experience
2. **Customize the Demos**: Modify the scripts to work with your own data
3. **Create Custom Analyzers**: Define field schemas for your specific use cases
4. **Integrate with Your Applications**: Use the client library in your own projects
5. **Explore Advanced Features**: Try the RAG integration and batch processing capabilities
6. **Evaluate Your Results**: Use the evaluation framework to assess extraction quality

