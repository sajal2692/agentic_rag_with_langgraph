# Agentic RAG with LangGraph

Exercises for the Agentic RAG live course for O'Reilly.

## Prerequisites

- Python 3.12+
- OpenAI API key
- Git

## Setup Instructions

### 1. Install uv (Python Package Manager)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

> **Note:** For detailed installation instructions and troubleshooting, visit the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/sajal2692/agentic_rag_with_langgraph.git
cd agentic_rag_with_langgraph

# Install dependencies
uv sync

# Create environment file
cp .env.example .env
```

### 3. Configure Environment Variables

Edit the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Setup Vector Databases

**Important:** You need to run BOTH ingestion scripts as different exercises use different database setups.

#### Step 1: Create Single Collection Database
```bash
# Run the single collection ingestion script
uv run python utils/ingest_data_in_single_collection.py
```

This creates a unified collection named `techmart_data` in `chroma_db/` (used by some exercises).

#### Step 2: Create Separate Collections Database
```bash
# Run the separate collections ingestion script
uv run python utils/ingest_data_in_separate_collections.py
```

This creates separate collections in `chroma_db_separate/` (used by other exercises):
- `techmart_catalog` (product catalog)
- `techmart_faq` (frequently asked questions)  
- `techmart_troubleshooting` (troubleshooting guides)

**Note:** Both scripts will show progress bars and take a few minutes to complete due to API calls to OpenAI for generating embeddings.

### 5. Verify Setup

After running both ingestion scripts, you should see:
- Progress bars showing data processing for each script
- Confirmation messages about successful ingestion
- ChromaDB files created in both `chroma_db/` and `chroma_db_separate/` directories

**Expected output structure:**
```
chroma_db/          # Single collection database
├── chroma.sqlite3
└── [collection-id-folders]/

chroma_db_separate/ # Separate collections database  
├── chroma.sqlite3
└── [collection-id-folders]/
```

## Project Structure

```
agentic_rag_with_langgraph/
├── data/                              # Source CSV files
│   ├── techmart_catalog.csv
│   ├── techmart_faq.csv
│   └── techmart_troubleshooting.csv
├── utils/                             # Utility scripts
│   ├── ingest_data_in_single_collection.py
│   └── ingest_data_in_separate_collections.py
├── chroma_db/                         # Single collection database (gitignored)
├── chroma_db_separate/                # Separate collections database (gitignored)
├── .env.example                       # Environment variables template
├── pyproject.toml                     # Project configuration
└── README.md
```

## Data Overview

The project uses TechMart e-commerce data:

- **Product Catalog** (64 products): Laptops, desktops, monitors, accessories with specifications
- **FAQ** (25 entries): Common customer questions about shipping, returns, warranties
- **Troubleshooting** (30 entries): Technical support guides for common issues

## Troubleshooting

### Common Issues

1. **uv command not found**
   - Make sure to restart your terminal after installing uv
   - On Windows, try using PowerShell instead of Command Prompt
   - Or use `python -m uv` instead of `uv`

2. **Windows execution policy errors**
   - Run PowerShell as Administrator
   - Use the full command: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

3. **OpenAI API errors**
   - Verify your API key is correct in `.env`
   - Check you have sufficient API credits

4. **Permission errors**
   - Windows: Run Command Prompt/PowerShell as Administrator
   - macOS/Linux: Try running with `sudo` if needed
   - Ensure you have write permissions in the project directory

5. **Import errors**
   - Run `uv sync` to ensure all dependencies are installed
   - Activate the virtual environment: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all setup steps were completed
3. Ask for help in the course discussion forum
