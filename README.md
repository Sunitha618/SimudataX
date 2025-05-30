# SimudataX

## Overview
Synthetic dataset generator using Apache Airflow and Groq open-source models, designed to generate and load data into MongoDB collections.

![image](https://github.com/user-attachments/assets/1ade1756-165d-4428-a587-1f4422824335)


## Project Core Values
- **Intelligent Data Generation**: Leveraging AI for synthetic dataset creation
- **Workflow Automation**: Streamlined data pipeline using Apache Airflow
- **Scalable Architecture**: Flexible data generation and loading mechanisms
- **Open-Source Transparency**: Modular design for easy customization

## Apache Airflow: Quick Introduction
Apache Airflow is an open-source platform for orchestrating complex computational workflows and data processing pipelines. Key benefits:
- Programmatic workflow definition
- Dependency management
- Monitoring and retry mechanisms
- Scalable task execution


### Airflow Installation
```bash
# Using pip
pip install apache-airflow

# Using Astro CLI (recommended)
brew install astro  # macOS
# Windows/Linux: Download from Astronomer website
```

## Prerequisites
- Python 3.8+
- Apache Airflow 2.x
- Groq API access
- MongoDB
- Astro CLI (optional)



## Configuration
1. Set Groq API credentials
2. Configure MongoDB connection
3. Define Airflow DAG parameters

## Project Structure
```
dataweave/
│
├── dags/                # Airflow workflow definitions
├── utils/               # Data processing utilities
│   ├── data_extraction.py
│   ├── data_loading.py
│   └── data_transformation.py
└── requirements.txt
```

## Data Extraction Strategy
- Use Groq AI models for intelligent data generation
- Parameterized generation based on predefined schemas
- Support for multiple data domain generations


## Custom Data Generation

### Modifying Prompt Class
To generate custom datasets, modify the `Prompt` class in `data_extraction.py`:

```python
class Prompt:
    def __init__(self):
        self.prompt = """
        [Your Custom Prompt Here]
        
        Ensure the prompt includes:
        - Structured data requirements
        - Specific domain context
        - Output format specifications
        """
```

### Prompt Design Guidelines
1. **Specificity**: Clearly define data structure
2. **Context**: Provide domain-specific details
3. **Format**: Specify JSON or desired output format
4. **Complexity**: Include nuanced generation requirements

### Example Customization Scenarios
- **Healthcare**: Patient record generation
- **Financial**: Transaction simulation
- **IoT**: Sensor data creation
- **Urban Planning**: Population movement modeling

### Best Practices
- Use clear, descriptive language
- Specify exact data fields
- Define constraints and uniqueness rules
- Include contextual parameters

### Prompt Engineering Tips
- Break down complex requirements
- Use JSON schema as a reference
- Provide example output structures
- Specify randomization or pattern requirements

## Advanced Customization
Extend the `Prompt` class with:
- Dynamic prompt generation methods
- Conditional data creation logic
- Domain-specific validation rules

## Example Use Case
```python
# Custom Prompt for E-commerce User Behavior
prompt.prompt = """
Generate synthetic user interaction data for an e-commerce platform...
"""
```

## Running the Project
```bash
# Start Airflow development environment
astro dev start
```

![image](https://github.com/user-attachments/assets/67f0853d-20c2-405e-8590-c2c0125e9f7d)

