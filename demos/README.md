# Yevrah Terminal Demos

This directory contains VHS tape files for generating demo GIFs showcasing Yevrah's capabilities.

## About VHS

[VHS](https://github.com/charmbracelet/vhs) is a tool for generating terminal GIFs. It uses "tape files" to script terminal sessions.

## Prerequisites

1. **Install VHS**:
   ```bash
   brew install vhs
   ```

2. **Set up API keys** in `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   COURTLISTENER_API_KEY=your_key_here
   COHERE_API_KEY=your_key_here  # Optional
   ```

3. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

## Demo Files

### demo1-natural-language.tape
Demonstrates a natural language query about premises liability cases in Florida.

**Query**: "customer slip fall store florida last 3 years"

**Showcases**:
- Natural language date parsing
- Jurisdiction mapping
- Dual search results (keyword + semantic)
- Full opinion analysis

### demo2-boolean-operators.tape
Demonstrates Boolean operator handling for precise searches.

**Query**: "employment AND discrimination, california"

**Showcases**:
- Boolean operator detection
- Automatic query optimization for dual search
- Separate keyword and semantic queries
- Result tagging by source

### demo3-jurisdiction-search.tape
Demonstrates jurisdiction-specific searches with date filters.

**Query**: "constitutional rights, supreme court, last 2 years"

**Showcases**:
- Court-specific searches
- Recent case filtering
- Supreme Court jurisdiction mapping

## Generating GIFs

To generate all demo GIFs:

```bash
cd demos
vhs demo1-natural-language.tape
vhs demo2-boolean-operators.tape
vhs demo3-jurisdiction-search.tape
```

Individual demo:
```bash
vhs demos/demo1-natural-language.tape
```

## Output

Generated GIFs will be saved in the `demos/` directory:
- `demo1-natural-language.gif`
- `demo2-boolean-operators.gif`
- `demo3-jurisdiction-search.gif`

## Customization

You can customize the tape files to:
- Change terminal theme (Dracula, Monokai, etc.)
- Adjust font size and dimensions
- Modify timing and delays
- Add different queries

See [VHS documentation](https://github.com/charmbracelet/vhs#vhs) for all available commands.

## Usage in README

Add the GIFs to your main README:

```markdown
## Demo

### Natural Language Search
![Natural Language Query Demo](demos/demo1-natural-language.gif)

### Boolean Operators
![Boolean Operators Demo](demos/demo2-boolean-operators.gif)

### Jurisdiction Search
![Jurisdiction Search Demo](demos/demo3-jurisdiction-search.gif)
```

## Notes

- **Recording requires real API calls** - Make sure you have valid API keys
- **Timing may vary** - Adjust `Sleep` commands based on your API response times
- **Terminal size** - Demos are optimized for 1200x800 resolution
- **Privacy** - Don't record with real sensitive data or API keys visible
