#!/usr/bin/env python3
"""
AI Memory System Demo - Intelligent Memory-Aware AI Generation
=============================================================

This demo showcases the advanced memory capabilities of our AI Creative Pipeline,
demonstrating intelligent memory-aware generation, semantic search, and context-aware prompting.

Features demonstrated:
• Memory reference detection using LLM analysis
• Semantic memory search and similarity matching
• Context-aware prompt processing
• Memory-driven creative generation
"""

import sys
import os
sys.path.append('app')

from core.memory_service import MemoryService
from core.llm_service import LocalLLMService

def print_header(title, char="="):
    """Print a formatted header"""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")

def print_section(title, char="-"):
    """Print a formatted section header"""
    print(f"\n{title}")
    print(char * len(title))

def main():
    print_header("🧠 AI MEMORY SYSTEM DEMONSTRATION")
    print("🚀 Intelligent Memory-Aware AI Creative Generation")
    print("✨ Advanced Semantic Memory Search & Context-Aware Processing")
    
    # Initialize services with correct memory path
    memory_service = MemoryService(persist_directory="app/datastore/memory")
    llm_service = LocalLLMService()
    
    # ===========================================
    # MEMORY SYSTEM STATUS
    # ===========================================
    print_section("📊 MEMORY SYSTEM STATUS")
    
    stats = memory_service.get_memory_stats()
    print(f"📚 Total Stored Memories: {stats['total_memories']}")
    print(f"🏷️  Unique Content Tags: {stats['unique_tags']}")
    
    if stats['top_tags']:
        popular_themes = ', '.join([f'{tag}({count})' for tag, count in stats['top_tags'][:5]])
        print(f"🎨 Popular Themes: {popular_themes}")
    
    # ===========================================
    # MEMORY-AWARE PROMPT ANALYSIS
    # ===========================================
    print_section("🔍 MEMORY-AWARE PROMPT ANALYSIS")
    
    test_scenarios = [
        {
            "name": "Memory Reference Detection Test",
            "prompt": "Generate a wooden toy car like the one I made last time",
            "description": "Testing detection of explicit past creation references"
        },
        {
            "name": "Simple Object Test", 
            "prompt": "A cute cartoon cat sitting down",
            "description": "Testing standard generation with simple objects"
        },
        {
            "name": "Memory-Aware Request Test",
            "prompt": "Create a red apple similar to my previous fruit",
            "description": "Testing contextual awareness with food items"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎯 Test {i}: {scenario['name']}")
        print(f"   📝 Prompt: \"{scenario['prompt']}\"")
        print(f"   🎪 Purpose: {scenario['description']}")
        
        # Memory reference detection
        memory_ref = llm_service.detect_memory_reference(scenario['prompt'])
        ref_status = "✅ DETECTED" if memory_ref['has_memory_reference'] else "❌ None"
        print(f"   🧠 Memory Reference: {ref_status}")
        
        if memory_ref['has_memory_reference']:
            print(f"      └─ Confidence: {memory_ref['confidence'].upper()}")
            if memory_ref.get('reference_type') and memory_ref['reference_type'] != 'none':
                print(f"      └─ Type: {memory_ref['reference_type']}")
        
        # Similar memory search
        similar_memories = memory_service.search_memories(scenario['prompt'], limit=3)
        print(f"   🔍 Similar Memories: {len(similar_memories)} found")
        
        if similar_memories:
            best_match = similar_memories[0]
            similarity = best_match.get('similarity', 0)
            print(f"      └─ Best Match: {similarity:.1%} similarity")
            print(f"      └─ \"{best_match['prompt'][:50]}...\"")
        
        # Processing decision
        if similar_memories or memory_ref['has_memory_reference']:
            print(f"   ⚡ Processing: MEMORY-AWARE generation")
        else:
            print(f"   ⚡ Processing: STANDARD generation")
    
    # ===========================================
    # SEMANTIC MEMORY SEARCH CAPABILITIES
    # ===========================================
    print_section("🔎 SEMANTIC MEMORY SEARCH CAPABILITIES")
    
    search_categories = [
        ("toy car", "Vehicles & Transportation"),
        ("cat", "Animals & Pets"),
        ("apple fruit", "Food & Natural Objects")
    ]
    
    for search_term, category in search_categories:
        print(f"\n🏷️  Category: {category}")
        print(f"   🔍 Search Term: \"{search_term}\"")
        
        results = memory_service.search_memories(search_term, limit=3)
        print(f"   📊 Results Found: {len(results)}")
        
        for j, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            prompt = result['prompt']
            print(f"      {j}. {similarity:.1%} - \"{prompt[:45]}...\"")
    
    # ===========================================
    # SYSTEM SUMMARY
    # ===========================================
    print_section("✅ SYSTEM CAPABILITIES SUMMARY")
    print("🧠 Intelligent Memory Detection:")
    print("   • Detects explicit references to past creations")
    print("   • Identifies variation and remake requests")
    print("   • Provides confidence levels and reasoning")
    print()
    print("🔍 Semantic Memory Search:")
    print("   • Finds thematically similar past creations")
    print("   • Ranks results by semantic similarity")
    print("   • Supports fuzzy matching across content types")
    print()
    print("⚡ Context-Aware Generation:")
    print("   • Adapts processing based on memory context")
    print("   • Provides memory-informed prompt enhancement")
    print("   • Maintains creative continuity across sessions")
    
    print_header("🚀 MEMORY SYSTEM READY FOR CREATIVE AI GENERATION", "=")
    print(f"📈 System Performance: {stats['total_memories']} memories indexed and searchable")
    print("💡 Ready to provide intelligent, memory-aware creative assistance!")

if __name__ == "__main__":
    main() 