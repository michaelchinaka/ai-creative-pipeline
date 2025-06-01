"""
AI Creative Pipeline - Streamlit Web Interface

This module provides a comprehensive web interface for the AI Creative Pipeline using Streamlit.
The interface allows users to interact with the AI system through a modern, intuitive web application
that supports prompt input, file browsing, memory management, and real-time visualization of results.

Key Features:
- Interactive prompt input with example suggestions
- File browser for generated content (images and 3D models)
- Memory browser for accessing past creations
- Real-time progress tracking during generation
- 3D model visualization with interactive viewer
- Responsive design with modern UI components

The application integrates with the core AI pipeline services including:
- Memory service for storing and retrieving past creations
- File management for organizing generated content
- Direct API communication with the backend pipeline

Requirements:
- Streamlit and associated UI components
- Access to the backend API server
- Memory service for contextual features
"""

import streamlit as st
import requests
import json
import base64
import time
from datetime import datetime
import os
from pathlib import Path
import ast
from streamlit_stl import stl_from_file
import streamlit.components.v1 as components
from streamlit_file_browser import st_file_browser

# Configure Streamlit page with metadata and layout settings
st.set_page_config(
    page_title="AI Creative Pipeline",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application header with branding and description
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
">
    <h1 style="
        color: white;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    ">AI Creative Pipeline</h1>
    <p style="
        color: rgba(255,255,255,0.9);
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 300;
    ">Transform your ideas into stunning visuals and interactive 3D models</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR CONFIGURATION
# Initialize session state variables for file management and user interactions

# Initialize session state for file selection management
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'clear_selection_flag' not in st.session_state:
    st.session_state.clear_selection_flag = False

# Memory Service Integration
# Import path configuration for accessing core services
import sys
import os
sys.path.append('app')

@st.cache_resource(show_spinner=False)
def get_memory_service():
    """
    Initialize and cache the memory service for persistent storage access.
    
    This function creates a cached instance of the MemoryService to avoid
    repeated initialization and improve application performance.
    
    Returns:
        MemoryService instance if successful, None if initialization fails
    """
    try:
        from core.memory_service import MemoryService
        return MemoryService(persist_directory="app/datastore/memory")
    except Exception as e:
        return None

# Initialize memory service with error handling
memory_service = get_memory_service()
memory_available = memory_service is not None

if not memory_available:
    st.sidebar.error("Memory service not available")

@st.cache_data(ttl=10, show_spinner=False)
def get_cached_memory_stats():
    """
    Retrieve memory statistics with caching for improved performance.
    
    This function caches memory statistics for 10 seconds to reduce
    database queries and improve UI responsiveness.
    
    Returns:
        Dictionary containing memory statistics or default values
    """
    if memory_available:
        return memory_service.get_memory_stats()
    return {'total_memories': 0}

@st.cache_data(ttl=10, show_spinner=False)
def get_cached_recent_memories(limit=5):
    """
    Retrieve recent memories with caching for improved performance.
    
    Args:
        limit: Maximum number of recent memories to retrieve
        
    Returns:
        List of recent memory objects or empty list if unavailable
    """
    if memory_available:
        return memory_service.get_recent_memories(limit=limit)
    return []

# MEMORY BROWSER SECTION
# Display memory statistics and provide access to recent creations
if memory_available:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Memory Browser")
    
    # Display memory statistics with caching for performance
    memory_stats = get_cached_memory_stats()
    total_memories = memory_stats.get('total_memories', 0)
    
    if total_memories > 0:
        st.sidebar.metric("Total Memories", total_memories)
        
        # Display recent memories with interactive selection
        recent_memories = get_cached_recent_memories(limit=5)
        if recent_memories:
            st.sidebar.markdown("**Recent Creations:**")
            
            # Container for compact memory display
            with st.sidebar.container():
                for i, memory in enumerate(recent_memories, 1):
                    prompt = memory['prompt']
                    # Truncate long prompts for compact display
                    display_prompt = prompt if len(prompt) <= 30 else prompt[:27] + "..."
                    
                    # Interactive button for memory selection
                    if st.sidebar.button(f'"{display_prompt}"', key=f"mem_{i}", help=f"Use: {prompt}"):
                        st.session_state.current_prompt = prompt
                        st.session_state.selected_prompt = prompt
                        st.rerun()
        else:
            st.sidebar.write("No recent memories found.")
    else:
        st.sidebar.info("No memories stored yet\nCreate something to build your memory!")

# FILE BROWSER SECTION
# Provide access to generated content with file management capabilities
st.sidebar.markdown("---")
st.sidebar.markdown("### File Browser") 

# Handle clear selection state to prevent immediate re-selection
if st.session_state.clear_selection_flag:
    st.session_state.clear_selection_flag = False
    # Reset file browser by changing its key to clear internal state
    browser_key = f"file_browser_{int(time.time())}"
else:
    browser_key = "file_browser"

# File browser implementation with content filtering
if os.path.exists("app/generated_content"):
    with st.sidebar:
        event = st_file_browser(
            path="app/generated_content",
            key=browser_key,
            show_choose_file=True,
            show_delete_file=False,
            show_new_folder=False,
            show_upload_file=False,
            show_preview=False,
            use_cache=False,
            extentions=['.png', '.jpg', '.jpeg', '.glb'],
            limit=50
        )
    
    # Handle file selection events with state management
    if event and isinstance(event, dict) and event.get('type') == 'SELECT_FILE' and not st.session_state.clear_selection_flag:
        target = event.get('target', {})
        relative_path = target.get('path')
        
        if relative_path:
            selected_path = os.path.join("app/generated_content", relative_path)
            st.session_state.selected_file = {
                'type': 'image' if selected_path.endswith(('.png', '.jpg', '.jpeg')) else '3d_model',
                'path': selected_path,
                'name': target.get('name', os.path.basename(selected_path)),
                'date': 'recent'
            }
    
    # Display file statistics for user awareness
    total_files = 0
    if os.path.exists("app/generated_content"):
        for root, dirs, files in os.walk("app/generated_content"):
            total_files += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.glb'))])
    
    if total_files > 0:
        st.sidebar.caption(f"Files available: {total_files}")
    
    # Display selected file information with management options
    if st.session_state.selected_file:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Selected:**")
        file_icon = "🖼️" if st.session_state.selected_file['type'] == 'image' else "🎭"
        st.sidebar.write(f"{file_icon} {st.session_state.selected_file['name']}")
        
        # Clear selection button with state management
        if st.sidebar.button("Clear Selection", key="clear_selection_btn"):
            st.session_state.selected_file = None
            st.session_state.clear_selection_flag = True
            st.rerun()
else:
    st.sidebar.info("No generated files yet")

# MAIN CONTENT AREA
# Primary interface for user interaction and content display
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Your Creative Prompt")
    
    # Initialize session state for prompt management
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    
    # Check if an example prompt was selected
    if 'selected_prompt' in st.session_state and st.session_state.selected_prompt:
        st.session_state.current_prompt = st.session_state.selected_prompt
        st.session_state.auto_submit = True
        st.session_state.pop('selected_prompt', None)
    
    # Input form
    with st.form("prompt_form"):
        user_prompt = st.text_area(
            "Enter your idea:",
            value=st.session_state.current_prompt,
            placeholder="A glowing dragon on a cliff at sunset...",
            height=100
        )
        
        submitted = st.form_submit_button("🚀 Generate", use_container_width=True)
        
        if user_prompt != st.session_state.current_prompt:
            st.session_state.current_prompt = user_prompt
    
    # Auto-submit logic
    if 'auto_submit' in st.session_state and st.session_state.auto_submit and st.session_state.current_prompt.strip():
        submitted = True
        st.session_state.pop('auto_submit', None)
    
    # Example prompts
    st.markdown("### 💡 Examples")
    
    # Check if we have memories to suggest memory-aware examples
    memory_aware_examples = []
    if memory_available and total_memories > 0:
        # Get some recent memories for context-aware examples
        recent_memories = memory_service.get_recent_memories(limit=3)
        if recent_memories:
            for memory in recent_memories:
                prompt = memory['prompt']
                if any(keyword in prompt.lower() for keyword in ['robot', 'dragon', 'city', 'crystal']):
                    if 'robot' in prompt.lower():
                        memory_aware_examples.append(f"Make a robot like the one I created before, but with wings")
                    elif 'dragon' in prompt.lower():
                        memory_aware_examples.append(f"Create a dragon similar to my last one, but this time in a forest")
                    elif 'city' in prompt.lower():
                        memory_aware_examples.append(f"Generate a city like my previous creation, but during sunrise")
                    break
    
    # Combine regular and memory-aware examples
    regular_examples = [
        "A cute cartoon cat sitting down",
        "A simple wooden chair",
        "A red apple on a table",
        "A small cactus in a pot",
        "A vintage camera"
    ]
    
    all_examples = memory_aware_examples + regular_examples
    
    for i, prompt in enumerate(all_examples[:5]):
        # Highlight memory-aware examples
        if i < len(memory_aware_examples):
            icon = "🧠✨"
            help_text = "Memory-aware prompt based on your past creations"
        else:
            icon = "✨"
            help_text = "Creative example prompt"
            
        if st.button(f"{icon} {prompt}", key=f"example_{i}", help=help_text):
            st.session_state.selected_prompt = prompt
            st.rerun()
    
    # How it works
    st.markdown("### ⚡ How it works")
    st.markdown("""
    1. **Enter** your creative idea
    2. **AI analyzes** your prompt  
    3. **Generates** an image
    4. **Creates** a 3D model
    5. **Browse** your creations in the sidebar
    """)

with col2:
    # Generation logic - check this first to override everything else during generation
    if submitted and st.session_state.current_prompt.strip():
        # Clear any selected file when starting new generation
        st.session_state.selected_file = None
        
        st.markdown("### 🔄 Generating...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Progressive status updates
            status_text.text("🧠 Analyzing prompt...")
            progress_bar.progress(15)
            time.sleep(4)
            
            status_text.text("✨ Enhancing prompt with AI...")
            progress_bar.progress(30)
            time.sleep(4)
            
            status_text.text("🎨 Generating image...")
            progress_bar.progress(50)
            
            payload = {"prompt": st.session_state.current_prompt.strip()}
            response = requests.post(
                "http://localhost:8888/execution",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            
            status_text.text("🎭 Converting to 3D model...")
            progress_bar.progress(85)
            time.sleep(0.5)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = ast.literal_eval(response.text)
                
                progress_bar.progress(100)
                status_text.text("")
                
                message = result.get('message', '')
                
                if "Complete Success" in message or "Steps 1, 2 & 2.2 Complete" in message:
                    st.success("🎉 Generation successful!")
                    
                    # Extract expanded prompt from the message - improved parsing
                    expanded_prompt = None
                    lines = message.split('\n')
                    
                    # Look for the enhanced prompt between specific markers
                    if "✨ Expanded Prompt:" in message and "🧠 LLM Analysis:" in message:
                        start_marker = "✨ Expanded Prompt:"
                        end_marker = "🧠 LLM Analysis:"
                        
                        start_idx = message.find(start_marker)
                        end_idx = message.find(end_marker)
                        
                        if start_idx != -1 and end_idx != -1:
                            # Extract text between markers
                            prompt_section = message[start_idx + len(start_marker):end_idx].strip()
                            
                            # Remove the <think> section if present
                            if "<think>" in prompt_section:
                                think_start = prompt_section.find("<think>")
                                think_end = prompt_section.find("</think>")
                                if think_start != -1 and think_end != -1:
                                    # Get text after </think>
                                    expanded_prompt = prompt_section[think_end + 8:].strip()
                            else:
                                expanded_prompt = prompt_section
                    
                    # Fallback to original parsing if the new method doesn't work
                    if not expanded_prompt:
                        # Look for the enhanced prompt more broadly
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            if any(keyword in line_lower for keyword in ["enhanced prompt:", "expanded prompt:", "improved prompt:"]):
                                # Try to get the prompt from the same line first
                                if ":" in line:
                                    prompt_part = line.split(":", 1)[-1].strip()
                                    if prompt_part and not prompt_part.startswith("Step"):
                                        expanded_prompt = prompt_part
                                        break
                                # If not found on same line, check next line
                                elif i + 1 < len(lines):
                                    next_line = lines[i + 1].strip()
                                    if next_line and not next_line.startswith("Step") and not next_line.startswith("Image"):
                                        expanded_prompt = next_line
                                        break
                            elif "step 1 complete" in line_lower and i + 1 < len(lines):
                                # Sometimes the enhanced prompt follows "Step 1 Complete"
                                next_line = lines[i + 1].strip()
                                if next_line and not next_line.startswith("Step") and not next_line.startswith("Image") and len(next_line) > 20:
                                    expanded_prompt = next_line
                                    break
                    
                    # Show expanded prompt if found
                    if expanded_prompt:
                        with st.expander("🧠 AI Enhanced Prompt", expanded=False):
                            st.info(expanded_prompt)
                    
                    # Extract and show memory information
                    memory_info = None
                    memory_reference_info = None
                    
                    # Check for memory reference detection
                    if "Memory Reference Detected:" in message:
                        lines = message.split('\n')
                        for line in lines:
                            if "Memory Reference Detected:" in line:
                                memory_reference_info = line.split("Memory Reference Detected:")[-1].strip()
                                break
                    
                    # Check for similar past creations
                    if "Similar past creations found:" in message:
                        lines = message.split('\n')
                        for i, line in enumerate(lines):
                            if "Similar past creations found:" in line:
                                # Extract the number
                                memory_count = line.split(":")[-1].strip().split()[0]
                                memory_info = f"Found {memory_count} memories related to your prompt"
                                break
                    
                    # Display memory information
                    if memory_reference_info:
                        st.info(f"🔍 {memory_reference_info}")
                    
                    if memory_info:
                        st.info(f"💭 {memory_info}")
                        
                        # Show memory connections if available
                        if "matches" in message:
                            with st.expander("🧠 Similar Past Creations", expanded=False):
                                lines = message.split('\n')
                                in_memory_section = False
                                memory_lines = []
                                
                                for line in lines:
                                    if "Similar past creations found:" in line:
                                        in_memory_section = True
                                        continue
                                    elif in_memory_section and line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                                        memory_lines.append(line.strip())
                                    elif in_memory_section and not line.strip().startswith(' ') and line.strip():
                                        break
                                
                                if memory_lines:
                                    for memory_line in memory_lines:
                                        st.write(f"📝 {memory_line}")
                    
                    # Extract file names - improved parsing
                    image_file = None
                    model_file = None
                    
                    for line in lines:
                        if "Image saved as:" in line:
                            image_file = line.split("Image saved as:")[-1].strip()
                        elif "3D model saved as:" in line:
                            model_file = line.split("3D model saved as:")[-1].strip()
                        elif "generated_content/" in line and (".png" in line or ".jpg" in line):
                            # Alternative parsing for image files
                            if not image_file:
                                parts = line.split()
                                for part in parts:
                                    if "generated_content/" in part and (".png" in part or ".jpg" in part):
                                        image_file = part
                                        break
                        elif "generated_content/" in line and (".glb" in line or ".obj" in line):
                            # Alternative parsing for model files
                            if not model_file:
                                parts = line.split()
                                for part in parts:
                                    if "generated_content/" in part and (".glb" in part or ".obj" in part):
                                        model_file = part
                                        break
                    
                    # Display results
                    tab1, tab2 = st.tabs(["🖼️ Image", "🎭 3D Model"])
                    
                    with tab1:
                        if image_file:
                            if image_file.startswith("generated_content"):
                                image_path = f"app/{image_file}"
                            else:
                                image_path = f"app/{image_file}"
                                today = datetime.now().strftime("%Y-%m-%d")
                                new_path = f"app/generated_content/{today}/{image_file}"
                                if os.path.exists(new_path):
                                    image_path = new_path
                            
                            if os.path.exists(image_path):
                                st.image(image_path, use_container_width=True)
                                with open(image_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download Image",
                                        data=file.read(),
                                        file_name=os.path.basename(image_file),
                                        mime="image/png"
                                    )
                            else:
                                st.error("Image not found")
                        else:
                            st.error("No image generated")
                    
                    with tab2:
                        if model_file:
                            # Handle multiple files in response
                            if " and " in model_file:
                                file_parts = model_file.split(" and ")
                                glb_file = None
                                for part in file_parts:
                                    if part.strip().endswith('.glb'):
                                        glb_file = part.strip()
                                        break
                                if glb_file:
                                    model_file = glb_file
                            
                            if model_file.startswith("generated_content"):
                                model_path = f"app/{model_file}"
                            else:
                                model_path = f"app/{model_file}"
                                today = datetime.now().strftime("%Y-%m-%d")
                                new_path = f"app/generated_content/{today}/{model_file}"
                                if os.path.exists(new_path):
                                    model_path = new_path
                            
                            if model_file.endswith('.glb'):
                                glb_path = model_path
                            else:
                                glb_path = model_path.replace('.obj', '.glb')
                            
                            if os.path.exists(glb_path):
                                try:
                                    with open(glb_path, 'rb') as f:
                                        model_data = f.read()
                                    model_base64 = base64.b64encode(model_data).decode()
                                    
                                    model_viewer_html = f"""
                                    <!DOCTYPE html>
                                    <html>
                                    <head>
                                        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
                                        <style>
                                            model-viewer {{
                                                width: 100%;
                                                height: 400px;
                                                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                                border-radius: 8px;
                                            }}
                                        </style>
                                    </head>
                                    <body>
                                        <model-viewer
                                            src="data:model/gltf-binary;base64,{model_base64}"
                                            alt="3D Model"
                                            auto-rotate
                                            camera-controls
                                            loading="eager">
                                        </model-viewer>
                                    </body>
                                    </html>
                                    """
                                    
                                    components.html(model_viewer_html, height=420)
                                    
                                    with open(glb_path, "rb") as file:
                                        st.download_button(
                                            label="📥 Download 3D Model",
                                            data=file.read(),
                                            file_name=os.path.basename(glb_path),
                                            mime="model/gltf-binary"
                                        )
                                        
                                except Exception as e:
                                    st.error("Error loading 3D model")
                            else:
                                st.error("3D model not found")
                        else:
                            st.error("No 3D model generated")
                
                else:
                    st.error("Generation failed")
                    st.write(message)
            
            else:
                st.error(f"Request failed: {response.status_code}")
        
        except requests.exceptions.Timeout:
            st.error("Request timed out")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Check if a file is selected (only show when NOT generating)
    elif 'selected_file' in st.session_state and st.session_state.selected_file:
        selected = st.session_state.selected_file
        
        st.markdown(f"### {selected['name']}")
        
        if selected['type'] == 'image':
            if os.path.exists(selected['path']):
                st.image(selected['path'], caption=selected['name'], use_container_width=True)
                
                with open(selected['path'], "rb") as file:
                    st.download_button(
                        label="📥 Download",
                        data=file.read(),
                        file_name=selected['name'],
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.error("Image not found")
        
        elif selected['type'] == '3d_model':
            if os.path.exists(selected['path']) and selected['name'].endswith('.glb'):
                try:
                    with open(selected['path'], 'rb') as f:
                        model_data = f.read()
                    model_base64 = base64.b64encode(model_data).decode()
                    
                    # Clean 3D viewer
                    model_viewer_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
                        <style>
                            model-viewer {{
                                width: 100%;
                                height: 500px;
                                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                border-radius: 8px;
                            }}
                        </style>
                    </head>
                    <body>
                        <model-viewer
                            src="data:model/gltf-binary;base64,{model_base64}"
                            alt="3D Model"
                            auto-rotate
                            camera-controls
                            loading="eager">
                        </model-viewer>
                    </body>
                    </html>
                    """
                    
                    components.html(model_viewer_html, height=520)
                    
                    with open(selected['path'], "rb") as file:
                        st.download_button(
                            label="📥 Download",
                            data=file.read(),
                            file_name=selected['name'],
                            mime="model/gltf-binary",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error("Error loading 3D model")
            else:
                st.error("3D model not found")
    
    else:
        # Default view (only show when NOT generating and no file selected)
        st.markdown("### 🎨 Create Something Amazing")
        st.markdown("""
        **Enter your creative prompt** and watch the AI transform your idea into:
        - ✨ A stunning image
        - 🎭 An interactive 3D model
        
        Your creations will appear here and be saved in the sidebar for future viewing.
        """)

