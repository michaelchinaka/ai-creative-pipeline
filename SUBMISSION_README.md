# 🚀 AI Creative Pipeline - Transform Ideas into Images and 3D Models

## 🎯 What This Does

This AI Creative Pipeline takes your text prompts and transforms them into:
1. **Enhanced Prompts** - Local LLM (DeepSeek) analyzes and expands your ideas
2. **Beautiful Images** - Generates high-quality images from enhanced prompts  
3. **Interactive 3D Models** - Converts images into 3D models you can view and download
4. **Smart Memory** - Remembers your creations and can reference them in future generations

**Example:** Type *"A glowing dragon on a cliff at sunset"* → Get an image + 3D model + stored in memory for future reference!

## 🏆 Bonus Features Implemented

✅ **Visual GUI with Streamlit** - Complete web interface with real-time progress, 3D viewing, and file management  
✅ **ChromaDB for Memory Similarity** - Semantic search using vector embeddings to find similar past creations  
✅ **Local Browser for 3D Assets** - Integrated file browser to explore, view, and download all generated content  

## 🧠 Memory Functionality (Core Feature)

The system implements both **Short-Term** and **Long-Term** memory as required:

### 📚 **Long-Term Memory (Persistent)**
- **Storage**: ChromaDB vector database with persistent storage in `app/datastore/memory/`
- **What's Stored**: Every creation includes prompt, expanded prompt, LLM analysis, image path, 3D model path, and timestamp
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- **Persistence**: Survives application restarts and system reboots


### **Session Context (Short-Term)**
- Active during single web session
- Tracks current generation progress
- Maintains UI state and file selections

### 🔍 **Memory Reference Detection**
The LLM actively detects when users reference past creations:
- **Explicit References**: *"like the one I made last time"*, *"similar to my dragon"*
- **Temporal References**: *"last Thursday"*, *"yesterday"*, *"before"*
- **Variation Requests**: *"but with wings"*, *"this time with crystals"*
- **Comparative Language**: *"similar to"*, *"like my previous"*

### 🎯 **Memory-Aware Processing**
When memory references are detected:
1. **Similarity Search**: Finds relevant past creations using semantic matching
2. **Context Integration**: LLM uses past creation context to enhance new prompts
3. **Smart Expansion**: Prompts are expanded with knowledge of user's creative history
4. **Confidence Scoring**: Each memory match includes similarity percentage

### 💡 **Example Memory Usage**
```
User: "Generate a robot like the one I made before, but with wings"

System Process:
1. 🔍 Detects memory reference: "like the one I made before"
2. 🔎 Searches past creations for robots
3. 📚 Finds: "A steampunk robot playing violin" (78% similarity)
4. ✨ Enhances prompt: "A steampunk robot with intricate brass wings, playing violin, vintage gears visible..."
5. 🎨 Generates image and 3D model using enhanced context
```

### 🧠 **Memory-Aware AI Processing**

The system intelligently detects when users reference past creations:

1. 🔍 **Detects**: "Make a cat like my last one but orange"
2. 📚 **Searches**: Previous cat creations (finds tabby cat, siamese cat)  
3. 📚 **Finds**: "A cute cartoon cat sitting down" (85% similarity)
4. ✨ **Enhances prompt**: "A cute cartoon cat sitting down with orange fur, round eyes, playful pose..."

### 🔧 **Technical Implementation**

- **ChromaDB**: Vector similarity search for memory matching
- **Sentence Transformers**: Semantic embeddings for prompt understanding  
- **Local LLM Integration**: DeepSeek for intelligent prompt expansion
- **Streamlit Interface**: Beautiful, responsive web UI

## 🚀 Quick Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Setup Ollama + DeepSeek
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull DeepSeek model (this will take a few minutes)
ollama pull deepseek-r1:1.5b

# Start Ollama service (keep this running)
ollama serve
```

### 3. Start the Backend API
```bash
cd app
./start.sh
```
**Keep this terminal running** - you'll see logs here showing the pipeline in action.

### 4. Launch the Web Interface
```bash
# In a new terminal
streamlit run streamlit_app.py
```

Open your browser to: **http://localhost:8501**

## 🎨 How to Generate Content

1. **Enter a Prompt** - Type your creative idea in the text area
2. **Click "🚀 Generate"** - Watch the progress as your idea comes to life
3. **View Results** - See both the generated image and interactive 3D model
4. **Browse Past Creations** - Use the sidebar to explore your memory bank

### Example Prompts to Try:
- *"A cute cartoon cat sitting down"*
- *"A simple wooden chair"*
- *"A red apple on a table"*
- *"Generate a cat like the one I made before, but orange"* (memory-aware!)




### Test Memory Functionality
```bash
cd app
source venv/bin/activate
PYTHONPATH=. python ../demo_for_submission.py
```
This shows memory detection, similarity matching, and context awareness in action.

## 📊 What You'll See in the Logs

When you generate content, watch the backend terminal for detailed logs showing:

```
🚀 Starting AI pipeline for prompt: 'A glowing dragon...'
🧠 Step 1: Understanding user intent with local LLM...
🔍 Memory reference detection: {'has_memory_reference': True, 'confidence': 'high'}
📚 Found 3 similar past creations for context
✨ Expanded prompt: A majestic glowing dragon with iridescent scales...
🎨 Step 2: Generating image from expanded prompt...
💾 Image saved as: generated_content/2024-01-15/image_20240115_143022.png
🎭 Step 2.2: Converting image to 3D model...
🎭 3D model saved as: generated_content/2024-01-15/model_20240115_143022.glb
🧠 Step 3: Storing creation in memory...
✅ Memory stored with ID: 12345
✅ All pipeline steps completed successfully!
```

## 📁 Generated Files

All your creations are saved in `app/generated_content/` organized by date:
- **Images**: `.png` files 
- **3D Models**: `.glb` files (viewable in the web interface)

## 🔧 System Architecture

```
User Prompt → DeepSeek LLM → Enhanced Prompt → Openfabric Text-to-Image → Image
     ↓                                                                      ↓
Memory Detection                                               Openfabric Image-to-3D
     ↓                                                                      ↓
Memory Storage ← ← ← ← ← Final Creation ← ← ← ← ← ← ← ← ← 3D Model
```

## 🚨 Troubleshooting

**"Ollama not responding"**
```bash
ollama serve  # Make sure this is running
ollama list   # Check DeepSeek model is installed
```

**"ModuleNotFoundError"**  
```bash
cd app
source venv/bin/activate  # Use the pre-configured environment
```

**"No generations appearing"**
- Check both terminals are running (backend + streamlit)
- Verify Ollama is running with `ollama list`
- Check logs in the backend terminal for specific errors

## 🎯 Key Features Demonstrated

- ✅ **End-to-End Pipeline**: Text → LLM Enhancement → Image → 3D Model
- ✅ **Memory-Aware Processing**: Detects and uses past creation context  
- ✅ **Real-Time Web Interface**: Beautiful Streamlit dashboard
- ✅ **Persistent Storage**: All creations saved and browsable
- ✅ **Interactive 3D Viewing**: View and download 3D models in browser

---

**Ready to create?** Follow the setup steps above and start generating amazing content! The system learns from your creations and gets smarter with every prompt. 