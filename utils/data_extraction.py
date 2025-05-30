import os
import random
import time
import logging
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_generator.log'),
        logging.StreamHandler()
    ]
)

# logger = logging.getLogger(__name__)
# logger.info("Starting Data extraction")

logger = logging.getLogger(__name__)
logger.info("Starting Data extraction")

class JSONExporter:
    """Class to handle JSON file operations"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = 'data/extracted_data'
        self._ensure_output_directory()

    def _ensure_output_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")

    def save_to_json(self, data, filename=None):
        """Save data to JSON file"""
        try:
            if filename is None:
                filename = f'data.json'
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert DataFrame to JSON structure
            if isinstance(data, pd.DataFrame):
                json_data = []
                for _, row in data.iterrows():
                    json_data.append({
                        'prompt': row['prompt'],
                        'response': row['response']
                    })
            else:
                json_data = data

            # Save with pretty printing
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Successfully saved data to {filepath}")
            return filepath
        
        except Exception as e:
            self.logger.error(f"Failed to save JSON file: {str(e)}", exc_info=True)
            raise

class DataGenerator:
    def __init__(self, api_key=None, model="llama3-70b-8192", temperature=0.5):
        """Initialize the generator with API credentials and parameters"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing DataGenerator")
        
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.logger.error("No API key found")
            raise ValueError("API key is required")
            
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.examples = []
        self.json_exporter = JSONExporter()
        self.logger.info(f"Initialized with model: {model}, temperature: {temperature}")

    def generate_and_save_examples(self, prompt, number_of_examples, filename=None):
        """Generate examples and save to JSON"""
        self.logger.info(f"Starting generation process for {number_of_examples} examples")
        
        try:
            # Generate and parse examples
            df = self.generate_and_parse_examples(prompt, number_of_examples)
            
            # Save to JSON
            json_path = self.json_exporter.save_to_json(df, filename)
            
            self.logger.info(f"Complete process finished. Data saved to {json_path}")
            return df, json_path
            
        except Exception as e:
            self.logger.error("Failed in generate_and_save_examples", exc_info=True)
            raise

    def generate_and_parse_examples(self, prompt, number_of_examples):
        """Generate and parse examples in one step"""
        self.logger.info(f"Starting generation and parsing of {number_of_examples} examples")
        
        try:
            # Generate examples
            self.generate_examples(prompt, number_of_examples)
            
            # Parse examples
            df = self.parse_examples()
            
            self.logger.info(f"Successfully generated and parsed {len(df)} unique examples")
            return df
            
        except Exception as e:
            self.logger.error("Failed to generate and parse examples", exc_info=True)
            raise

    def generate_single_example(self, prompt, prev_examples):
        """Generate a single example using Groq API"""
        self.logger.info("Generating single example")
        max_retries = 3
        retry_delay = 1
        retry_count = 0

        while retry_count < max_retries:
            try:
                messages = self._prepare_messages(prompt, prev_examples)
                response = self._call_api(messages)
                
                if self._is_valid_response(response):
                    self.logger.info("Successfully generated example")
                    return response
                
                self.logger.warning(f"Invalid response format. Retry {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error generating example: {str(e)}", exc_info=True)
                retry_count += 1
                time.sleep(retry_delay)
        
        self.logger.error("Maximum retries exceeded. Failed to generate example")
        return None

    def _prepare_messages(self, prompt, prev_examples):
        """Prepare messages for API call"""
        self.logger.debug("Preparing messages for API call")
        messages = [
            {
                "role": "user",
                "content": f'Now, generate a prompt/response pair for `{prompt}`. Do so in the exact format requested:\n```\n<prompt>prompt</prompt>\n<response>response_goes_here</response>\n```\n\nOnly one prompt/response pair should be generated per turn. Never miss the <prompt> & <response> tags'
            }
        ]

        if prev_examples:
            sample_size = min(10, len(prev_examples))
            self.logger.debug(f"Including {sample_size} previous examples")
            sample_examples = random.sample(prev_examples, sample_size)
            for example in sample_examples:
                messages.append({"role": "assistant", "content": example})
                messages.append({"role": "user", "content": 'Now, generate another prompt/response pair. Make it unique.'})
        
        return messages

    def _call_api(self, messages):
        """Make the API call to Groq"""
        self.logger.debug("Making API call to Groq")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
            )
            self.logger.debug("API call successful")
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise

    def _is_valid_response(self, response):
        """Check if the response contains required tags"""
        is_valid = response and '<prompt>' in response and '<response>' in response
        if not is_valid:
            self.logger.warning("Invalid response format detected")
        return is_valid

    def generate_examples(self, prompt, number_of_examples):
        """Generate multiple examples"""
        self.logger.info(f"Starting generation of {number_of_examples} examples")
        self.examples = []
        
        for i in range(number_of_examples):
            self.logger.info(f'Generating example {i + 1}/{number_of_examples}')
            example = self.generate_single_example(prompt, self.examples)
            if example:
                self.examples.append(example)
                self.logger.debug(f"Example {i + 1} generated successfully")
            else:
                self.logger.error(f"Failed to generate example {i + 1}")
        
        self.logger.info(f"Completed generating examples. Success rate: {len(self.examples)}/{number_of_examples}")
        return self.examples

    def parse_examples(self):
        """Parse examples into a DataFrame"""
        self.logger.info("Parsing examples into DataFrame")
        prompts = []
        responses = []

        for i, example in enumerate(self.examples):
            try:
                prompt, response = self._extract_prompt_response(example)
                prompts.append(prompt)
                responses.append(response)
                self.logger.debug(f"Successfully parsed example {i + 1}")
            except (ValueError, IndexError) as e:
                self.logger.error(f"Error parsing example {i + 1}: {str(e)}", exc_info=True)
                continue

        df = pd.DataFrame({
            'prompt': prompts,
            'response': responses
        })
        df_unique = df.drop_duplicates()
        self.logger.info(f"Created DataFrame with {len(df_unique)} unique examples")
        return df_unique

    def _extract_prompt_response(self, example):
        try:
            prompt_start = example.index('<prompt>') + len('<prompt>')
            prompt_end = example.index('</prompt>')
            response_start = example.index('<response>') + len('<response>')
        
            # Check if '</response>' exists
            if '</response>' in example:
                response_end = example.index('</response>')
                prompt = example[prompt_start:prompt_end].strip()
                response = example[response_start:response_end].strip()
                return prompt, response
            else:
                # Log a warning and skip if '</response>' tag is missing
                logger.warning("Example missing '</response>' tag; skipping example.")
                return None, None

        except ValueError as e:
            logger.error(f"Error extracting prompt/response: {str(e)}")
            raise



class Prompt:
    """Class to handle the prompt"""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Prompt")
        self.prompt = """

        Generate structured data for a disaster management system that dynamically allocates resources in response to a crisis event, ensuring no duplicates. 
        The output should include the following components, formatted as JSON: 

        1. input_data: - event_reports: An array of objects detailing various reports from multiple sources, including: - source (e.g., 'weather_station', 
        'emergency_calls', 'social_media') - timestamp (ISO 8601 format) - data: An object containing relevant metrics such as: - For weather stations: rainfall_mm, 
        wind_speed_kmh, pressure_hpa - For emergency calls: call_volume, flood_reports, rescue_requests - For social media: flood_mentions, evacuation_requests, 
        damage_reports - geospatial_data: Information about the affected area, including: - affected_area (type and coordinates in a Polygon format) - population_density 
        (total population and distribution across urban, suburban, and rural areas) - critical_infrastructure: An array of objects detailing critical facilities 
        (type, location, and relevant capacity). - resource_inventory: An array detailing available resources, including type, availability, 
        and capacity. - historical_context: Information about similar past events and flood-prone areas, including their risk levels. 
       
        2. output_data: - incident_id: A unique identifier for the incident. - timestamp: The timestamp of the output data. - crisis_type: The type of crisis 
        (e.g., hurricane, flood). - severity_score: A score reflecting the severity of the situation (0-10 scale). - affected_area: Details about the geographical impact,
        including center coordinates and radius. - population_at_risk: Estimated number of individuals at risk. - current_status: A description of the current 
        situation and key threats. - resource_recommendations: An array of resource needs with quantities, priorities, and deployment locations. - priority_actions: An array of recommended actions, 
        their descriptions, responsible agencies, and estimated execution times. - potential_escalations: Scenarios for possible escalation with probabilities, 
        impact scores, and additional resource needs. - data_sources_used: A list of data sources that informed the output. - confidence_score: A score indicating the 
        confidence level of the assessment (0-1 scale). Ensure the generated data is realistic and reflects the complexities involved in disaster management, 
        while avoiding duplicates.
        
        """.strip()
        self.logger.debug("Prompt initialized with content")


# def main():
    
    
#     try:
#         # Initialize classes
#         prompt = Prompt()
#         generator = DataGenerator()
        
#         # Generate examples and save to JSON
#         df, json_path = generator.generate_and_save_examples(
#             prompt=prompt.prompt,
#             number_of_examples=5,
#             filename="menu_examples.json"
#         )
        
#         # Display results
#         logger.info(f'Generated {len(df)} unique examples')
#         print(f'\nGenerated {len(df)} unique examples. Here are the first few:')
#         print(df.head())
#         print(f'\nData saved to: {json_path}')
        
#         logger.info("Main execution completed successfully")
#         return df, json_path
    
#     except Exception as e:
#         logger.error("Main execution failed", exc_info=True)
#         raise


# if __name__ == "__main__":
#     main()