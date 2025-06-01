"""
AI Creative Pipeline - Main Execution Module

This module implements the core AI Creative Pipeline that transforms user prompts into 
images and 3D models through a three-step process:
1. Local LLM analysis and prompt enhancement
2. Text-to-image generation via Openfabric
3. Image-to-3D model conversion via Openfabric
4. Memory storage for future reference

The pipeline includes intelligent memory-aware processing that can detect references
to past creations and enhance new prompts based on historical context.
"""

import logging
from typing import Dict
import base64
import os
from datetime import datetime

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub
from core.llm_service import LocalLLMService
from core.memory_service import MemoryService

# Global configuration storage for user-specific settings
configurations: Dict[str, ConfigClass] = dict()

# Initialize core services
# These services handle LLM communication and memory management
llm_service = LocalLLMService()
memory_service = MemoryService()

# Openfabric Application IDs
# These are the verified working app IDs for the required services
TEXT_TO_IMAGE_APP_ID = "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"  # Text-to-image generation
IMAGE_TO_3D_APP_ID = "f0b5f319156c4819b9827000b17e511a.node3.openfabric.network"    # Image-to-3D model conversion


def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Callback function to store user-specific configuration data.
    
    This function is called by the Openfabric framework to provide user configurations
    that contain app IDs and other settings specific to each user.

    Args:
        configuration: Dictionary mapping user IDs to their configuration objects
        state: Current application state (unused in this implementation)
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id: '{uid}'")
        configurations[uid] = conf


def execute(model: AppModel) -> None:
    """
    Main execution entry point for the AI Creative Pipeline.
    
    This function implements the complete pipeline that takes a user prompt and transforms
    it through multiple stages to produce both an image and a 3D model, while maintaining
    memory of past creations for context-aware generation.
    
    Pipeline Steps:
    1. Prompt Analysis: Use local LLM to analyze and enhance the user prompt
    2. Memory Integration: Check for references to past creations and integrate context
    3. Image Generation: Create image using enhanced prompt via Openfabric
    4. 3D Conversion: Convert generated image to 3D model via Openfabric
    5. Memory Storage: Store the creation for future reference and learning
    
    Args:
        model: AppModel containing request data and response structure
    """
    # Extract user input from the model
    request: InputClass = model.request
    user_prompt = request.prompt

    logging.info(f"Starting AI pipeline for prompt: '{user_prompt}'")

    # Retrieve or create user configuration
    user_config: ConfigClass = configurations.get('super-user', None)
    if not user_config:
        # Create default configuration with verified app IDs
        user_config = ConfigClass()
        user_config.app_ids = [TEXT_TO_IMAGE_APP_ID, IMAGE_TO_3D_APP_ID]
    
    logging.info(f"User config: {user_config}")

    # Initialize Openfabric service stub with configured app IDs
    app_ids = user_config.app_ids if user_config.app_ids else [TEXT_TO_IMAGE_APP_ID, IMAGE_TO_3D_APP_ID]
    logging.info(f"🔗 Initializing Stub with app IDs: {app_ids}")
    
    try:
        stub = Stub(app_ids)
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize Stub: {e}")
        response: OutputClass = model.response
        response.message = f"Error: Failed to connect to Openfabric apps: {str(e)}"
        return

    # ------------------------------
    # STEP 1: UNDERSTAND THE USER 🧠
    # Use local LLM (DeepSeek) to interpret and expand prompts
    # ------------------------------

    try:
        logging.info("🧠 Step 1: Understanding user intent with local LLM...")
        
        # Verify LLM service availability
        if not llm_service.is_available():
            logging.error("❌ Local LLM service is not available")
            response: OutputClass = model.response
            response.message = "Error: Local LLM service is not available. Please ensure Ollama is running with DeepSeek model."
            return
        
        # STEP 1A: Memory Reference Detection
        # Analyze the prompt to detect references to past creations
        memory_reference = llm_service.detect_memory_reference(user_prompt)
        logging.info(f"🔍 Memory reference detection: {memory_reference}")
        
        # Log LLM reasoning for debugging and transparency
        if memory_reference.get('explanation'):
            logging.info(f"🔍 LLM reasoning: {memory_reference['explanation']}")
        
        # STEP 1B: Semantic Memory Search
        # Search for similar past creations to provide context
        similar_creations = memory_service.find_similar_creations(user_prompt, limit=5)
        if similar_creations:
            logging.info(f"📚 Found {len(similar_creations)} similar past creations for context")
            for i, creation in enumerate(similar_creations, 1):
                similarity = creation.get('similarity', 0)
                logging.info(f"  {i}. \"{creation['prompt']}\" ({similarity:.1%} similar)")
        else:
            logging.info("📚 No similar past creations found")
        
        # STEP 1C: Context-Aware Prompt Processing
        # Choose processing method based on available memory context
        if similar_creations or memory_reference['has_memory_reference']:
            logging.info("🧠 Using memory-aware intent analysis...")
            user_analysis = llm_service.interpret_memory_aware_intent(user_prompt, similar_creations)
            
            # Use memory-aware prompt expansion
            logging.info("✨ Using memory-aware prompt expansion...")
            expanded_prompt = llm_service.expand_prompt_with_memory(user_prompt, similar_creations)
        else:
            logging.info("🧠 Using standard intent analysis...")
            user_analysis = llm_service.interpret_user_intent(user_prompt)
            expanded_prompt = llm_service.expand_prompt(user_prompt)
        
        logging.info(f"📊 User intent analysis: {user_analysis}")
        logging.info(f"✨ Expanded prompt: {expanded_prompt}")
        logging.info("✅ Step 1 completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error in Step 1: {e}")
        response: OutputClass = model.response
        response.message = f"Error in Step 1 (LLM Processing): {str(e)}"
        return

    # ------------------------------
    # STEP 2: TEXT TO IMAGE 🎨
    # Use Openfabric Text-to-Image app
    # ------------------------------

    try:
        logging.info("🎨 Step 2: Generating image from expanded prompt...")
        
        # Use the working text-to-image app ID
        logging.info(f"🔗 Using Text-to-Image app: {TEXT_TO_IMAGE_APP_ID}")
        
        # Call the text-to-image service with the enhanced prompt
        image_result = stub.call(TEXT_TO_IMAGE_APP_ID, {'prompt': expanded_prompt}, 'super-user')
        logging.info(f"📸 Image generation result keys: {image_result.keys() if image_result else 'None'}")
        
        if not image_result:
            raise Exception("No result from Text-to-Image app")
        
        # Extract the image data (this might be in different formats - base64, bytes, etc.)
        image_data = None
        if 'result' in image_result:
            image_data = image_result['result']
        elif 'image' in image_result:
            image_data = image_result['image']
        elif 'output' in image_result:
            image_data = image_result['output']
        
        if not image_data:
            raise Exception("No image data found in result")
        
        # Create organized file structure for generated content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized output directory structure
        output_dir = "generated_content"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create date-based subdirectory for better organization
        date_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)
        
        image_filename = os.path.join(date_dir, f"image_{timestamp}.png")
        
        # Handle different image data formats that the API might return
        if isinstance(image_data, bytes):
            # Direct binary data - save as is
            with open(image_filename, 'wb') as f:
                f.write(image_data)
        elif isinstance(image_data, str):
            # May be base64 encoded or raw string
            try:
                image_bytes = base64.b64decode(image_data)
                with open(image_filename, 'wb') as f:
                    f.write(image_bytes)
            except:
                # If not base64, save as text file for debugging
                with open(f"{image_filename}.txt", 'w') as f:
                    f.write(str(image_data))
        
        logging.info(f"💾 Image saved as: {image_filename}")
        logging.info("✅ Step 2 completed successfully!")
        
        # STEP 3: IMAGE-TO-3D MODEL CONVERSION
        # Convert the generated image into a 3D model using Openfabric
        try:
            logging.info("Step 3: Converting image to 3D model...")
            
            # Use the verified image-to-3D application ID
            logging.info(f"Using Image-to-3D app: {IMAGE_TO_3D_APP_ID}")
            
            # Read the generated image to send to the 3D conversion service
            with open(image_filename, 'rb') as f:
                image_bytes = f.read()
            
            # Encode image as base64 for JSON serialization
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Call the image-to-3D service with the generated image
            # Note: The exact input format may need adjustment based on the app's schema
            model_3d_result = stub.call(IMAGE_TO_3D_APP_ID, {'input_image': image_base64}, 'super-user')
            logging.info(f"3D model generation result keys: {model_3d_result.keys() if model_3d_result else 'None'}")
            
            if not model_3d_result:
                raise Exception("No result from Image-to-3D app")
            
            # Extract 3D model data from the service response
            # The response format may vary, so we check multiple possible keys
            model_3d_data = None
            if 'result' in model_3d_result:
                model_3d_data = model_3d_result['result']
            elif 'model' in model_3d_result:
                model_3d_data = model_3d_result['model']
            elif 'output' in model_3d_result:
                model_3d_data = model_3d_result['output']
            elif 'mesh' in model_3d_result:
                model_3d_data = model_3d_result['mesh']
            elif 'generated_object' in model_3d_result:
                model_3d_data = model_3d_result['generated_object']
            elif 'video_object' in model_3d_result:
                model_3d_data = model_3d_result['video_object']
            
            if not model_3d_data:
                raise Exception("No 3D model data found in result")
            
            # Save the 3D model in GLB format (glTF Binary)
            model_3d_filename_glb = os.path.join(date_dir, f"model_{timestamp}.glb")
            
            # Handle different 3D model data formats that the API might return
            if isinstance(model_3d_data, bytes):
                # Direct binary data - save as GLB file
                with open(model_3d_filename_glb, 'wb') as f:
                    f.write(model_3d_data)
            elif isinstance(model_3d_data, str):
                # May be base64 encoded or raw model data
                try:
                    model_bytes = base64.b64decode(model_3d_data)
                    # Save as GLB format (binary)
                    with open(model_3d_filename_glb, 'wb') as f:
                        f.write(model_bytes)
                except:
                    # If not base64, save as text with GLB extension
                    # Note: The API typically returns GLB format
                    with open(model_3d_filename_glb, 'w') as f:
                        f.write(str(model_3d_data))
            
            logging.info(f"3D model saved as: {model_3d_filename_glb}")
            logging.info("Step 3 completed successfully!")
            
            # STEP 4: MEMORY STORAGE AND SIMILARITY ANALYSIS
            # Store this creation in memory for future context-aware generation
            try:
                logging.info("Step 4: Storing creation in memory...")
                
                # Store the creation in the memory system for future reference
                logging.info("Memory System: Storing creation in memory...")
                try:
                    memory_id = memory_service.store_creation(
                        prompt=user_prompt,
                        expanded_prompt=expanded_prompt,
                        llm_analysis=user_analysis.get('analysis', ''),
                        image_file=image_filename,
                        model_file=model_3d_filename_glb,
                        execution_id=str(id(model))  # Use model object ID as unique execution ID
                    )
                    
                    if memory_id:
                        logging.info(f"Memory stored with ID: {memory_id}")
                        
                        # We already found similar creations in Step 1, so use those results
                        if similar_creations:
                            logging.info(f"Found {len(similar_creations)} similar past creations (from Step 1)")
                            for i, creation in enumerate(similar_creations, 1):
                                similarity = creation.get('similarity', 0)
                                prompt = creation.get('prompt', 'Unknown prompt')
                                logging.info(f"  {i}. \"{prompt}\" ({similarity:.1%} similar)")
                        else:
                            logging.info("No similar past creations found")

                except Exception as e:
                    logging.error(f"Memory storage failed: {e}")
                    memory_id = None

                # Prepare comprehensive response with memory awareness
                memory_text = ""
                memory_awareness_text = ""
                
                # Add memory reference detection information
                if memory_reference.get('has_memory_reference'):
                    memory_awareness_text = f"\nMemory Reference Detected: {memory_reference.get('explanation', 'LLM detected reference to past creations')}"
                    if memory_reference.get('reference_type') and memory_reference['reference_type'] != 'none':
                        memory_awareness_text += f" (Type: {memory_reference['reference_type']})"
                
                # Add memory storage results
                if memory_id:
                    memory_text = f"\nMemory System: Creation stored for future recall (ID: {memory_id})"
                    if similar_creations:
                        memory_text += f"\nSimilar past creations found: {len(similar_creations)} matches"
                        for i, creation in enumerate(similar_creations[:3], 1):  # Show top 3 matches
                            similarity = creation.get('similarity', 0)
                            memory_text += f"\n   {i}. \"{creation['prompt']}\" ({similarity:.1%} similar)"
                    else:
                        memory_text += "\nNo similar past creations found (this might be your first creation with this theme)"

                # Prepare final success response with all pipeline results
                response: OutputClass = model.response
                response.message = f"""AI Pipeline Complete Success!

Original Prompt: {user_prompt}
{memory_awareness_text}
Expanded Prompt: {expanded_prompt}
LLM Analysis: {user_analysis.get('analysis', 'No analysis available')}

Step 1 Complete: Image generated and saved to: {image_filename}
Step 2 Complete: 3D Model generated and saved to: {model_3d_filename_glb}
Step 3 Complete: GLB format saved to: {model_3d_filename_glb}
{memory_text}

All pipeline steps completed successfully!
Files saved in: {date_dir}
Ready for the next creation!
"""

            except Exception as e:
                logging.error(f"Error in Step 4: {e}")
                similar_info = "\n\nWarning: Memory system error, but creation was successful"
            
        except Exception as e:
            logging.error(f"Error in Step 3: {e}")
            # Still return Step 1 results even if Step 3 fails
            response: OutputClass = model.response
            response.message = f"""AI Pipeline - Step 1 Complete, Step 3 Failed

Original Prompt: {user_prompt}

LLM Analysis: {user_analysis.get('analysis', 'No analysis available')}

Expanded Prompt: {expanded_prompt}

Image-to-3D Conversion Error: {str(e)}

Note: Step 1 (LLM) completed successfully, but Step 3 (Image-to-3D) failed."""
            return
    except Exception as e:
        logging.error(f"Error in Step 2: {e}")
        # Still return Step 1 results even if Step 2 fails
        response: OutputClass = model.response
        response.message = f"""AI Pipeline - Step 1 Complete, Step 2 Failed

Original Prompt: {user_prompt}

LLM Analysis: {user_analysis.get('analysis', 'No analysis available')}

Expanded Prompt: {expanded_prompt}

Image Generation Error: {str(e)}

Note: Step 1 (LLM) completed successfully, but Step 2 (Text-to-Image) failed."""
        return