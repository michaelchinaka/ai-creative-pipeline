#!/usr/bin/env python3
"""
Memory System Demo - Test the AI memory capabilities
This script demonstrates how the memory system works for the AI Creative Pipeline
"""

import sys
import os
sys.path.append('app')

from core.memory_service import MemoryService
from core.llm_service import LocalLLMService

def main():
    print("🧠 AI Memory System Demo")
    print("=" * 50)
    
    # Initialize services
    memory_service = MemoryService(persist_directory="app/datastore/memory")
    llm_service = LocalLLMService()
    
    # Get current memory stats
    stats = memory_service.get_memory_stats()
    print(f"📊 Current memory stats:")
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Unique tags: {stats['unique_tags']}")
    
    if stats['top_tags']:
        print(f"   Top tags: {', '.join([f'{tag}({count})' for tag, count in stats['top_tags'][:5]])}")
    
    print("\n" + "=" * 50)
    
    # Test memory-aware prompts
    test_prompts = [
        "A steampunk robot with glowing eyes",
        "Generate a robot like the one I made last time but with wings",
        "Create a cyberpunk city at night",
        "Make something similar to my dragon, but this time with crystals",
        "A magical forest with floating gems"
    ]
    
    print("🔍 Testing Memory-Aware Prompt Processing:")
    print("-" * 40)
    
    for prompt in test_prompts:
        print(f"\n🎯 Testing prompt: \"{prompt}\"")
        
        # Step 1: Detect memory references
        memory_ref = llm_service.detect_memory_reference(prompt)
        print(f"   🔍 Memory reference detected: {memory_ref['has_memory_reference']}")
        print(f"   🎯 Confidence: {memory_ref['confidence']}")
        if memory_ref.get('explanation'):
            print(f"   💭 LLM reasoning: {memory_ref['explanation']}")
        if memory_ref.get('reference_type') and memory_ref['reference_type'] != 'none':
            print(f"   🏷️ Reference type: {memory_ref['reference_type']}")
        
        # Step 2: Search for similar memories
        similar_memories = memory_service.search_memories(prompt, limit=3)
        print(f"   📚 Similar memories found: {len(similar_memories)}")
        
        for i, memory in enumerate(similar_memories, 1):
            similarity = memory.get('similarity', 0)
            print(f"      {i}. \"{memory['prompt'][:50]}...\" ({similarity:.1%} similar)")
        
        # Step 3: Show what memory-aware processing would do
        if similar_memories:
            print(f"   🧠 Would use MEMORY-AWARE processing (found {len(similar_memories)} similar memories)")
        elif memory_ref['has_memory_reference']:
            print(f"   🧠 Would use MEMORY-AWARE processing (detected memory reference)")
        else:
            print(f"   🤖 Would use STANDARD processing (no memory context)")
    
    print("\n" + "=" * 50)
    print("🎯 Memory Search Demo:")
    print("-" * 30)
    
    # Test semantic search
    search_queries = [
        "robot",
        "dragon",
        "cyberpunk city",
        "magical creatures",
        "steampunk"
    ]
    
    for query in search_queries:
        results = memory_service.search_memories(query, limit=2)
        print(f"\n🔍 Search: \"{query}\" → {len(results)} results")
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            prompt = result['prompt']
            tags = result.get('tags', [])
            print(f"   {i}. \"{prompt[:40]}...\" ({similarity:.1%})")
            if tags:
                print(f"      Tags: {', '.join(tags[:5])}")
    
    print("\n" + "=" * 50)
    print("✅ Memory System Demo Complete!")
    print(f"💡 Your AI system has {stats['total_memories']} memories stored")
    print("💡 It can detect memory references in user prompts")
    print("💡 It provides context-aware generation based on past creations")
    print("💡 Users can search and recall past creations semantically")

if __name__ == "__main__":
    main() 