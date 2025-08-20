# Azure AI Content Understanding Demo

This demo package provides comprehensive examples and utilities for working with Azure AI Content Understanding service. It demonstrates document analysis, audio processing, and custom analyzer creation capabilities.

## 🚀 Quick Start

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

4. **Run the comprehensive demo**:
```bash
python comprehensive_demo.py
```

## 📁 Package Structure

```
demo/
├── requirements.txt              # Python dependencies
├── .env.template                # Environment configuration template
├── content_understanding_client.py   # Core client library
├── document_analysis_demo.py     # Document analysis examples
├── audio_analysis_demo.py        # Audio analysis examples
├── healthcare_assessment_demo.py # Healthcare assessment automation
├── healthcare_utils.py          # Healthcare compliance utilities
├── golf_coaching_assessment_demo.py # Golf coaching streaming analysis
├── golf_coaching_utils.py       # Golf analytics and streaming utilities
├── validate_golf_demo.py        # Golf demo validation script
├── utils.py                     # Utility functions
├── comprehensive_demo.py        # Main demo script
├── batch_processing_demo.py     # Batch processing examples
├── rag_integration_demo.py      # RAG integration examples
└── README.md                    # This file
```

## 🎯 Demo Scripts

### 1. Comprehensive Demo (`comprehensive_demo.py`)
The main demo script that showcases all capabilities:
- Configuration validation
- Document analysis with prebuilt analyzers
- Custom analyzer creation and usage
- Audio processing and transcription
- Analyzer management operations

**Run it:**
```bash
python comprehensive_demo.py
```

### 2. Document Analysis Demo (`document_analysis_demo.py`)
Focused on document processing capabilities:
- PDF, image, and Office document analysis
- Custom field extraction
- Document structure analysis

**Run it:**
```bash
python document_analysis_demo.py
```

### 3. Audio Analysis Demo (`audio_analysis_demo.py`)
Demonstrates audio processing features:
- Audio transcription and diarization
- Call center analysis
- Meeting audio processing
- Custom audio analyzer creation

**Run it:**
```bash
python audio_analysis_demo.py
```

### 4. Batch Processing Demo (`batch_processing_demo.py`)
Shows how to process multiple files efficiently:
- Concurrent processing
- Progress tracking
- Error handling and retry logic

**Run it:**
```bash
python batch_processing_demo.py
```

### 5. RAG Integration Demo (`rag_integration_demo.py`)
Demonstrates integration with RAG (Retrieval-Augmented Generation) systems:
- Content extraction for knowledge bases
- Azure AI Search integration
- Vector embedding preparation

**Run it:**
```bash
python rag_integration_demo.py
```

### 6. Healthcare Assessment Demo (`healthcare_assessment_demo.py`)
Shows automated patient assessment form completion from care staff-patient conversations:
- Patient assessment automation
- Clinical data extraction
- HIPAA-compliant processing
- Multiple assessment types (mental health, physical health, geriatric)

**Run it:**
```bash
python healthcare_assessment_demo.py
```

### 7. Golf Coaching Assessment Demo (`golf_coaching_assessment_demo.py`)
Demonstrates real-time streaming audio analysis for golf coaching conversations:
- Real-time question-answer detection from streaming audio
- Automatic assessment form completion
- Golf-specific performance analysis
- Multiple skill level scenarios (beginner, intermediate, advanced)
- Coaching recommendations and practice plans

**Run it:**
```bash
python golf_coaching_assessment_demo.py
```

**Features:**
- Real-time streaming audio processing
- Automatic question-answer detection from conversations
- Golf-specific assessment forms and field mapping
- Performance analysis and skill level assessment
- Coaching recommendations and structured practice plans
- Multiple player scenarios (beginner to advanced)

## 🔧 Configuration
- Clinical alerts and recommendations
- HIPAA compliance utilities

## 🔧 Configuration

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

## 📊 Sample Data

The demos use publicly available sample files:

- **Documents**: PDF invoices, receipts, contracts
- **Audio**: Call recordings, meeting audio
- **Images**: Charts, diagrams, scanned documents

You can replace these with your own files by updating the URLs in the demo scripts.

## 🎨 Features Demonstrated

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

## 🏥 Healthcare Use Case

The healthcare assessment demo demonstrates automated patient assessment form completion from care staff-patient conversations. This showcases real-world application of Content Understanding in clinical settings.

### Features

- **Multiple Assessment Types**: Mental health screening, physical health assessment, geriatric assessment
- **Clinical Data Extraction**: Automatically extracts relevant clinical information from conversations
- **HIPAA Compliance**: Built-in utilities for data anonymization and secure handling
- **Confidence Scoring**: Validates extraction confidence for clinical decision support
- **Clinical Alerts**: Flags high-risk conditions requiring immediate attention
- **EHR Integration**: Export capabilities to standard healthcare formats (HL7 FHIR)

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

### Healthcare Utilities (`healthcare_utils.py`)

- **HIPAAComplianceHelper**: Data anonymization and audit logging
- **ClinicalDataValidator**: Assessment validation and consistency checking
- **Assessment Templates**: Expandable library for different specialties
- **EHR Export**: Standard healthcare format conversion

### Usage Example

```python
from healthcare_assessment_demo import PatientAssessmentAnalyzer

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

## 🏌️ Golf Coaching Use Case

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

### Golf Coaching Utilities (`golf_coaching_utils.py`)

- **StreamingAudioAnalyzer**: Real-time audio processing and conversation flow analysis
- **GolfPerformanceAnalyzer**: Performance insights and skill level assessment
- **PlayerProfile**: Comprehensive golfer profile management
- **Practice Plan Generator**: Structured practice recommendations based on assessment

### Usage Example

```python
from golf_coaching_assessment_demo import GolfAssessmentAnalyzer

# Initialize analyzer
analyzer = GolfAssessmentAnalyzer()

# Start streaming assessment session
session_id = await analyzer.start_streaming_assessment(
    athlete_id="GOLFER_001",
    coach_id="COACH_001",
    session_notes="Initial assessment for intermediate player"
)

# Process streaming audio (in real implementation)
# analyzer.add_audio_stream_chunk(audio_chunk)

# Stop session and get results
result = await analyzer.stop_streaming_assessment()
print(f"Assessment completed: {result['success']}")
```

## 📝 Output

## 📝 Output

Demo results are saved to the `./output` directory:

```
output/
├── demo_report.json              # Overall demo summary
├── prebuilt_analysis_result.json # Prebuilt analyzer results
├── custom_analysis_result.json   # Custom analyzer results
├── audio_analysis_result.json    # Audio processing results
├── healthcare_assessments/       # Healthcare assessment results
│   ├── patient_assessment_001.json
│   └── validation_report_001.json
├── golf_assessments/             # Golf coaching assessment results
│   ├── golf_assessment_session_001.json
│   └── performance_analysis_001.json
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

## 🔗 Additional Resources

- [Azure AI Content Understanding Documentation](../README.md)
- [REST API Reference](https://docs.microsoft.com/en-us/rest/api/contentunderstanding/)
- [Service Limits and Quotas](../service-limits.md)
- [Language and Region Support](../language-region-support.md)

## 💡 Next Steps

1. **Customize the Demos**: Modify the scripts to work with your own data
2. **Create Custom Analyzers**: Define field schemas for your specific use cases
3. **Integrate with Your Applications**: Use the client library in your own projects
4. **Explore Advanced Features**: Try the RAG integration and batch processing capabilities

## 🤝 Contributing

This demo is part of the Azure AI Content Understanding documentation. If you find issues or have suggestions for improvements, please contribute back to the documentation repository.

## 📄 License

This demo code is provided as part of the Azure AI services documentation and samples.
