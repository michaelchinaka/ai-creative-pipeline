"""
Local Large Language Model Service

This module provides a service interface for interacting with local LLMs via Ollama.
It handles prompt interpretation, expansion, and creative enhancement with memory-aware
capabilities for contextual generation based on past creations.

The service supports both standard and memory-aware operations:
- Standard prompt expansion for image generation
- Memory-aware prompt enhancement using similar past creations
- Intent analysis for understanding user requests
- Memory reference detection for context-aware responses

Requires Ollama to be running locally with the DeepSeek model installed.
"""

import logging
import ollama
from typing import Optional, List, Dict, Any


class LocalLLMService:
    """
    Service for interacting with local Large Language Models via Ollama.
    
    This service provides intelligent prompt processing capabilities including:
    - Prompt expansion for better image generation
    - Memory-aware enhancement using historical context
    - User intent analysis and interpretation
    - Detection of references to past creations
    
    The service is designed to work with the DeepSeek model but can be configured
    to use other compatible models available through Ollama.
    
    Attributes:
        model_name (str): The name of the LLM model to use
        client (ollama.Client): The Ollama client instance for API communication
    """
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        """
        Initialize the LocalLLMService with the specified model.
        
        Args:
            model_name: The name of the Ollama model to use for LLM operations.
                       Defaults to "deepseek-r1:1.5b" which provides good performance
                       for creative tasks while being resource-efficient.
        """
        self.model_name = model_name
        self.client = ollama.Client()
        logging.info(f"Initialized LocalLLMService with model: {model_name}")
    
    def expand_prompt_with_memory(self, user_prompt: str, similar_memories: List[Dict[str, Any]] = None) -> str:
        """
        Expand and enhance a user prompt using historical context from similar past creations.
        
        This method takes a basic user prompt and expands it into a rich, detailed description
        suitable for image generation. It leverages similar past creations to provide context
        and inspiration while ensuring the output remains unique and creative.
        
        Args:
            user_prompt: The original user prompt to expand
            similar_memories: List of similar past creations containing prompts, tags,
                            and similarity scores to use as context
            
        Returns:
            Enhanced and expanded prompt optimized for image generation, incorporating
            contextual elements from similar past creations while maintaining uniqueness
        """
        try:
            # Build memory context if available
            memory_context = ""
            if similar_memories:
                memory_context = "\n\nRELEVANT PAST CREATIONS:\n"
                for i, memory in enumerate(similar_memories[:3], 1):
                    similarity = memory.get('similarity', 0)
                    memory_context += f"{i}. \"{memory['prompt']}\" (similarity: {similarity:.1%})\n"
                    if memory.get('tags'):
                        memory_context += f"   Tags: {', '.join(memory['tags'])}\n"
                
                memory_context += "\nUse these past creations as inspiration and context, but create something new and unique.\n"

            system_prompt = f"""You are a creative AI assistant specialized in expanding simple prompts into vivid, detailed descriptions for image generation. 

Your task is to take a basic prompt and expand it into a rich, descriptive prompt that will generate stunning visuals. Focus on:
- Visual details (colors, lighting, textures, composition)
- Artistic style and mood
- Environmental context
- Technical photography/art terms when appropriate

{memory_context}

Keep the expansion focused and under 200 words. Return ONLY the expanded prompt, no explanations."""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Expand this prompt for image generation: {user_prompt}"}
                ]
            )
            
            expanded_prompt = response['message']['content'].strip()
            logging.info(f"Memory-aware expansion - Original: {user_prompt}")
            logging.info(f"Memory-aware expansion - Enhanced: {expanded_prompt}")
            
            return expanded_prompt
            
        except Exception as e:
            logging.error(f"Error in memory-aware prompt expansion: {e}")
            # Fallback to regular expansion if memory-aware processing fails
            return self.expand_prompt(user_prompt)

    def expand_prompt(self, user_prompt: str) -> str:
        """
        Expand and enhance a user prompt for improved image generation results.
        
        This method takes a basic user prompt and transforms it into a detailed,
        vivid description that will produce better results when used with image
        generation models. It adds visual details, artistic style information,
        and technical terms while maintaining the core intent of the original prompt.
        
        Args:
            user_prompt: The original user prompt to expand
            
        Returns:
            Enhanced and expanded prompt optimized for image generation
        """
        try:
            system_prompt = """You are a creative AI assistant specialized in expanding simple prompts into vivid, detailed descriptions for image generation. 

Your task is to take a basic prompt and expand it into a rich, descriptive prompt that will generate stunning visuals. Focus on:
- Visual details (colors, lighting, textures, composition)
- Artistic style and mood
- Environmental context
- Technical photography/art terms when appropriate

Keep the expansion focused and under 200 words. Return ONLY the expanded prompt, no explanations."""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Expand this prompt for image generation: {user_prompt}"}
                ]
            )
            
            expanded_prompt = response['message']['content'].strip()
            logging.info(f"Original prompt: {user_prompt}")
            logging.info(f"Expanded prompt: {expanded_prompt}")
            
            return expanded_prompt
            
        except Exception as e:
            logging.error(f"Error expanding prompt: {e}")
            # Fallback to original prompt if LLM expansion fails
            return user_prompt
    
    def interpret_memory_aware_intent(self, user_prompt: str, similar_memories: List[Dict[str, Any]] = None) -> dict:
        """
        Analyze user prompt to understand intent with awareness of past creations.
        
        This method provides deeper analysis by considering the user's history of
        past creations. It can identify patterns, variations, and relationships
        between the current request and previous work, enabling more contextual
        and personalized creative assistance.
        
        Args:
            user_prompt: The user's input prompt to analyze
            similar_memories: List of similar past creations for contextual analysis
            
        Returns:
            Dictionary containing comprehensive analysis including:
            - subject: Main subject or object being created
            - style: Artistic style or mood
            - setting: Environment or background context
            - intent: What the user wants to create
            - memory_connection: How this relates to past creations
            - variation_type: Type of variation if building on past work
        """
        try:
            # Build memory context for analysis
            memory_context = ""
            memory_analysis = ""
            
            if similar_memories:
                memory_context = "\n\nPAST CREATIONS CONTEXT:\n"
                for i, memory in enumerate(similar_memories[:3], 1):
                    similarity = memory.get('similarity', 0)
                    memory_context += f"{i}. \"{memory['prompt']}\" ({similarity:.1%} similar)\n"
                    if memory.get('tags'):
                        memory_context += f"   Style/Tags: {', '.join(memory['tags'])}\n"
                
                memory_analysis = f"""

Also analyze:
- How this request relates to past creations
- Whether it's asking for variations, improvements, or completely new ideas
- What elements from past creations might be relevant"""

            analysis_prompt = f"""Analyze this creative prompt and extract key information. Return a JSON-like response with:
- subject: main subject/object
- style: artistic style or mood
- setting: environment or background
- intent: what the user wants to create
- memory_connection: how this relates to past creations (if any)
- variation_type: if this is a variation of past work (remix, evolution, new take, etc.)

{memory_context}

Prompt to analyze: {user_prompt}{memory_analysis}"""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            return {
                "original_prompt": user_prompt,
                "analysis": response['message']['content'].strip(),
                "confidence": "high",
                "memory_aware": len(similar_memories) > 0 if similar_memories else False,
                "similar_count": len(similar_memories) if similar_memories else 0
            }
            
        except Exception as e:
            logging.error(f"Error in memory-aware intent analysis: {e}")
            # Fallback to standard intent analysis if memory-aware processing fails
            return self.interpret_user_intent(user_prompt)
    
    def interpret_user_intent(self, user_prompt: str) -> dict:
        """
        Analyze user prompt to understand intent and extract key creative elements.
        
        This method performs standard analysis of user prompts to understand what
        the user wants to create. It extracts key elements like subject matter,
        artistic style, and creative intent without considering historical context.
        
        Args:
            user_prompt: The user's input prompt to analyze
            
        Returns:
            Dictionary containing analysis results including:
            - subject: Main subject or object
            - style: Artistic style or mood
            - setting: Environment or background
            - intent: What the user wants to create
        """
        try:
            analysis_prompt = f"""Analyze this creative prompt and extract key information. Return a JSON-like response with:
- subject: main subject/object
- style: artistic style or mood
- setting: environment or background
- intent: what the user wants to create

Prompt to analyze: {user_prompt}"""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            # For now, return a simple analysis
            # In a production system, you'd parse the JSON response
            return {
                "original_prompt": user_prompt,
                "analysis": response['message']['content'].strip(),
                "confidence": "high"
            }
            
        except Exception as e:
            logging.error(f"Error interpreting user intent: {e}")
            return {
                "original_prompt": user_prompt,
                "analysis": "Unable to analyze prompt",
                "confidence": "low"
            }
    
    def detect_memory_reference(self, user_prompt: str) -> dict:
        """
        Detect if the user is referencing past creations in their prompt using reliable keyword detection.
        
        Args:
            user_prompt (str): The user's input prompt
            
        Returns:
            dict: Information about memory references found
        """
        try:
            # First, use reliable keyword detection
            prompt_lower = user_prompt.lower()
            reliable_keywords = [
                "like the one i made", "like i made", "like last time", "similar to my", 
                "like my", "my previous", "last time", "but this time", "remake",
                "like before", "based on my", "new version", "like that one"
            ]
            
            keyword_detected = False
            detected_phrase = ""
            for keyword in reliable_keywords:
                if keyword in prompt_lower:
                    keyword_detected = True
                    detected_phrase = keyword
                    break
            
            # If keywords detected, return immediately
            if keyword_detected:
                logging.info(f"Memory reference detected via keyword: '{detected_phrase}'")
                return {
                    "has_memory_reference": True,
                    "confidence": "high",
                    "explanation": f"Found explicit memory reference: '{detected_phrase}'",
                    "reference_type": "time_reference" if "time" in detected_phrase else "variation"
                }
            
            # For ambiguous cases, use LLM but with stricter parsing
            analysis_prompt = f"""Analyze this prompt for memory references. Answer ONLY with YES or NO.

Look for these EXACT phrases in the prompt:
- "like the one I made" / "like I made"
- "similar to my" / "like my"  
- "last time" / "before"
- "but this time" / "new version"
- "remake" / "based on my"

Prompt: "{user_prompt}"

Does this prompt contain ANY of these exact phrases? Answer YES or NO only."""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            response_text = response['message']['content'].strip().upper()
            
            # Strict parsing - only accept clear YES/NO
            if "YES" in response_text and "NO" not in response_text:
                has_reference = True
                confidence = "medium"
                explanation = "LLM detected memory reference"
            elif "NO" in response_text and "YES" not in response_text:
                has_reference = False
                confidence = "high"
                explanation = "No memory reference detected"
            else:
                # Default to no reference for ambiguous responses
                has_reference = False
                confidence = "low"
                explanation = "Ambiguous LLM response, defaulting to no reference"
            
            logging.info(f"LLM-based detection: {has_reference} - {explanation}")
            
            return {
                "has_memory_reference": has_reference,
                "confidence": confidence,
                "explanation": explanation,
                "reference_type": "variation" if has_reference else "none",
                "llm_response": response_text
            }
            
        except Exception as e:
            logging.error(f"Error in memory reference detection: {e}")
            # Fallback to keyword detection only
            prompt_lower = user_prompt.lower()
            reliable_keywords = [
                "like the one i made", "like i made", "like last time", "similar to my", 
                "like my", "my previous", "last time", "but this time", "remake"
            ]
            has_reference = any(keyword in prompt_lower for keyword in reliable_keywords)
            
            return {
                "has_memory_reference": has_reference,
                "confidence": "medium",
                "explanation": f"Keyword detection: {'Found' if has_reference else 'No'} memory indicators",
                "reference_type": "variation" if has_reference else "none"
            }
    
    def is_available(self) -> bool:
        """
        Check if the LLM service is available and responsive.
        
        Returns:
            bool: True if service is available, False otherwise
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logging.error(f"LLM service not available: {e}")
            return False 