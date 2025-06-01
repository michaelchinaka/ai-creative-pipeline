"""
AI Memory Service for Creative Generation

This module provides a memory system for storing and retrieving creative AI generations
using ChromaDB for persistent storage and semantic search capabilities. The service
enables users to find past creations using natural language queries and maintains
context across creative sessions.

Key Features:
- Semantic search using sentence transformers
- Persistent storage with ChromaDB
- Automatic tag extraction from prompts
- Memory-aware content retrieval
- Temporal organization of creations

The service stores comprehensive metadata about each creation including original prompts,
expanded prompts, LLM analysis, generated files, and automatically extracted tags.
"""

import chromadb
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import os
import uuid

class MemoryService:
    """
    AI Memory Service using ChromaDB for semantic search and memory management.
    
    This service provides persistent storage and retrieval of creative AI generations
    with semantic search capabilities. It maintains a database of user creations that
    can be searched using natural language queries, enabling context-aware generation
    and memory-based creative assistance.
    
    The service uses sentence transformers for generating embeddings and ChromaDB
    for persistent storage with semantic similarity search capabilities.
    
    Attributes:
        persist_directory (str): Directory path for persistent storage
        client (chromadb.PersistentClient): ChromaDB client for database operations
        collection (chromadb.Collection): ChromaDB collection for storing memories
        encoder (SentenceTransformer): Sentence transformer for generating embeddings
    """
    
    def __init__(self, persist_directory: str = "datastore/memory"):
        """
        Initialize the memory service with ChromaDB and sentence transformers.
        
        Sets up the persistent storage directory, initializes the ChromaDB client,
        creates or retrieves the memory collection, and loads the sentence transformer
        model for generating embeddings.
        
        Args:
            persist_directory: Directory path to persist the ChromaDB database.
                             Defaults to "datastore/memory" for organized storage.
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists for persistent storage
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection for memories
        self.collection = self.client.get_or_create_collection(
            name="ai_creations",
            metadata={"description": "User creative AI generations with semantic search"}
        )
        
        # Initialize sentence transformer for embeddings
        # Using 'all-MiniLM-L6-v2' for good balance of speed and quality
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        existing_count = self.collection.count()
        logging.info(f"Memory Service initialized with {existing_count} existing memories")
    
    def store_creation(self, 
                      prompt: str, 
                      expanded_prompt: str,
                      llm_analysis: str,
                      image_file: str,
                      model_file: str,
                      execution_id: str) -> str:
        """
        Store a new creation in memory with semantic embeddings for future retrieval.
        
        This method stores comprehensive information about a creative generation session,
        including the original prompt, expanded prompt, LLM analysis, generated files,
        and automatically extracted tags. It creates semantic embeddings for efficient
        similarity-based retrieval.
        
        Args:
            prompt: Original user prompt that initiated the creation
            expanded_prompt: LLM-enhanced version of the original prompt
            llm_analysis: Analysis and interpretation from the LLM service
            image_file: Path to the generated image file
            model_file: Path to the generated 3D model file
            execution_id: Unique identifier for this execution session
            
        Returns:
            Unique memory ID for the stored creation, or None if storage failed
        """
        try:
            # Generate unique memory identifier
            memory_id = str(uuid.uuid4())
            
            # Extract semantic tags from prompts and analysis
            tags = self._extract_tags(prompt, expanded_prompt, llm_analysis)
            
            # Create comprehensive text for embedding generation
            # This combines all textual information for semantic search
            embedding_text = f"""
            Original: {prompt}
            Expanded: {expanded_prompt}
            Analysis: {llm_analysis}
            Tags: {', '.join(tags)}
            """
            
            # Generate semantic embedding for similarity search
            embedding = self.encoder.encode(embedding_text).tolist()
            
            # Create comprehensive metadata for storage
            metadata = {
                "prompt": prompt,
                "expanded_prompt": expanded_prompt,
                "llm_analysis": llm_analysis,
                "image_file": image_file,
                "model_file": model_file,
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "tags": json.dumps(tags),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S")
            }
            
            # Store in ChromaDB with embedding and metadata
            self.collection.add(
                embeddings=[embedding],
                documents=[embedding_text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            logging.info(f"Stored creation in memory: {memory_id} - '{prompt[:50]}...'")
            return memory_id
            
        except Exception as e:
            logging.error(f"Error storing creation in memory: {e}")
            return None
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories using semantic similarity to find relevant past creations.
        
        This method performs semantic search across stored memories using natural
        language queries. It generates embeddings for the query and finds the most
        similar stored creations based on semantic meaning rather than keyword matching.
        
        Args:
            query: Natural language search query describing what to find
            limit: Maximum number of results to return, defaults to 5
            
        Returns:
            List of matching memories with metadata, sorted by similarity score
        """
        try:
            # Generate embedding for the search query
            query_embedding = self.encoder.encode(query).tolist()
            
            # Perform semantic search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results into structured memory objects
            memories = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    memory = {
                        "id": results['ids'][0][i],
                        "prompt": metadata.get('prompt', ''),
                        "expanded_prompt": metadata.get('expanded_prompt', ''),
                        "llm_analysis": metadata.get('llm_analysis', ''),
                        "image_file": metadata.get('image_file', ''),
                        "model_file": metadata.get('model_file', ''),
                        "timestamp": metadata.get('timestamp', ''),
                        "tags": json.loads(metadata.get('tags', '[]')),
                        "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                        "date": metadata.get('date', ''),
                        "time": metadata.get('time', '')
                    }
                    memories.append(memory)
            
            logging.info(f"Found {len(memories)} memories for query: '{query}'")
            return memories
            
        except Exception as e:
            logging.error(f"Error searching memories: {e}")
            return []
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recently created memories for display and context.
        
        This method returns the most recent creations sorted by timestamp,
        useful for showing recent activity and providing quick access to
        latest work in user interfaces.
        
        Args:
            limit: Maximum number of memories to return, defaults to 10
            
        Returns:
            List of recent memories sorted by creation time (most recent first)
        """
        try:
            # Retrieve all memories with metadata
            all_results = self.collection.get(
                include=["metadatas"],
                limit=limit
            )
            
            memories = []
            if all_results['metadatas']:
                for i, metadata in enumerate(all_results['metadatas']):
                    memory = {
                        "id": all_results['ids'][i],
                        "prompt": metadata.get('prompt', ''),
                        "image_file": metadata.get('image_file', ''),
                        "model_file": metadata.get('model_file', ''),
                        "timestamp": metadata.get('timestamp', ''),
                        "tags": json.loads(metadata.get('tags', '[]')),
                        "date": metadata.get('date', ''),
                        "time": metadata.get('time', '')
                    }
                    memories.append(memory)
                
                # Sort by timestamp (most recent first)
                memories.sort(key=lambda x: x['timestamp'], reverse=True)
                memories = memories[:limit]
            
            logging.info(f"Retrieved {len(memories)} recent memories")
            return memories
            
        except Exception as e:
            logging.error(f"Error getting recent memories: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            total_count = self.collection.count()
            
            # Get all memories to analyze
            all_results = self.collection.get(include=["metadatas"])
            
            # Analyze tags
            all_tags = []
            dates = []
            
            if all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    tags = json.loads(metadata.get('tags', '[]'))
                    all_tags.extend(tags)
                    dates.append(metadata.get('date', ''))
            
            # Count tag frequency
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Get top tags
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get date range
            dates = [d for d in dates if d]
            date_range = {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            }
            
            stats = {
                "total_memories": total_count,
                "top_tags": top_tags,
                "date_range": date_range,
                "unique_tags": len(tag_counts)
            }
            
            logging.info(f"📊 Memory stats: {total_count} total memories, {len(tag_counts)} unique tags")
            return stats
            
        except Exception as e:
            logging.error(f"❌ Error getting memory stats: {e}")
            return {"total_memories": 0, "top_tags": [], "date_range": {}, "unique_tags": 0}
    
    def _extract_tags(self, prompt: str, expanded_prompt: str, llm_analysis: str) -> List[str]:
        """
        Extract semantic tags from prompts and analysis.
        
        Args:
            prompt: Original prompt
            expanded_prompt: Expanded prompt
            llm_analysis: LLM analysis
            
        Returns:
            List of extracted tags
        """
        # Common creative keywords to look for
        creative_keywords = [
            # Objects
            'robot', 'dragon', 'castle', 'forest', 'city', 'mountain', 'ocean', 'space', 'planet',
            'car', 'ship', 'building', 'tree', 'flower', 'animal', 'bird', 'fish', 'creature',
            'sword', 'shield', 'crown', 'gem', 'crystal', 'magic', 'spell', 'potion',
            
            # Styles
            'cyberpunk', 'steampunk', 'fantasy', 'sci-fi', 'medieval', 'futuristic', 'vintage',
            'modern', 'ancient', 'mystical', 'magical', 'dark', 'bright', 'colorful', 'glowing',
            'metallic', 'wooden', 'stone', 'glass', 'crystal',
            
            # Environments
            'sunset', 'sunrise', 'night', 'day', 'storm', 'rain', 'snow', 'desert', 'jungle',
            'underwater', 'sky', 'clouds', 'stars', 'moon', 'sun',
            
            # Emotions/Moods
            'peaceful', 'dramatic', 'mysterious', 'epic', 'serene', 'chaotic', 'beautiful',
            'terrifying', 'majestic', 'elegant', 'powerful', 'delicate'
        ]
        
        # Combine all text
        all_text = f"{prompt} {expanded_prompt} {llm_analysis}".lower()
        
        # Extract matching keywords
        tags = []
        for keyword in creative_keywords:
            if keyword in all_text:
                tags.append(keyword)
        
        # Add some basic categorization
        if any(word in all_text for word in ['robot', 'cyberpunk', 'futuristic', 'sci-fi']):
            tags.append('sci-fi')
        if any(word in all_text for word in ['dragon', 'magic', 'fantasy', 'medieval', 'castle']):
            tags.append('fantasy')
        if any(word in all_text for word in ['nature', 'forest', 'tree', 'flower', 'mountain']):
            tags.append('nature')
        if any(word in all_text for word in ['city', 'building', 'urban', 'street']):
            tags.append('urban')
        
        # Remove duplicates and limit
        tags = list(set(tags))[:10]
        
        return tags
    
    def find_similar_creations(self, reference_prompt: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find creations similar to a reference prompt.
        
        Args:
            reference_prompt: The prompt to find similar creations for
            limit: Maximum number of similar creations to return
            
        Returns:
            List of similar creations
        """
        return self.search_memories(reference_prompt, limit) 