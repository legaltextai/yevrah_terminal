"""
Main entry point for the AI-Enabled Research Assistant.
A conversational terminal app that helps litigators find relevant case law.
"""
import sys
import re
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from llm_client import LLMClient
from courtlistener import CourtListenerClient

# Suppress HTTP request logging from httpx (used by Groq)
logging.getLogger("httpx").setLevel(logging.WARNING)
from formatter import (
    display_results,
    display_error,
    display_api_url,
    display_assistant_message,
    display_search_info
)
from jurisdictions import (
    FEDERAL_APPELLATE_COURTS,
    STATE_SUPREME_COURTS,
    POPULAR_JURISDICTIONS,
    format_popular_jurisdictions
)

console = Console()

# Maximum results to display
MAX_RESULTS = 10


def display_welcome():
    """Display the welcome banner."""
    welcome_text = """
[bold cyan]━━━ Yevrah ━━━[/bold cyan]
[dim]AI-Enabled Research Assistant[/dim]

[magenta]✨ Dual Search Strategy with AI Reranking[/magenta]
[dim]• Runs both Keyword AND Semantic searches in parallel
• Fetches 20 results from each (40 total)
• Combines, deduplicates, and reranks with Cohere AI
• Shows you the top 10 most relevant cases[/dim]

[dim]Commands: 'exit' to quit | 'new' to reset | 'help' for options | 'jurisdictions' for court codes[/dim]
    """
    console.print(Panel(
        welcome_text,
        title="[bold cyan]⚖️[/bold cyan]",
        border_style="cyan",
        padding=(0, 2)
    ))

    # Display legal disclaimer
    disclaimer_text = """
[yellow]⚠️  LEGAL DISCLAIMER[/yellow]

This is an AI-enabled research assistant provided for informational purposes only.
It does not constitute legal advice. Always independently verify case citations, holdings,
and legal principles before relying on them. Use your professional judgment in evaluating
the relevance and applicability of search results to your specific situation.
    """
    console.print(Panel(
        disclaimer_text,
        border_style="yellow",
        padding=(0, 2)
    ))
    console.print()


def display_help():
    """Display help information."""
    help_text = """
[bold cyan]How Yevrah Works[/bold cyan]

[magenta]✨ Natural Language → Dual Search → AI Reranking[/magenta]

Just describe your legal issue naturally, including [bold]jurisdiction and time parameters[/bold].
Yevrah will automatically:

[green]1. Formulate TWO queries[/green] - One for keyword search, one for semantic search
[green]2. Parse parameters[/green] - Extract jurisdiction and dates from natural language
[green]3. Run BOTH searches[/green] - Keyword AND semantic in parallel (20 results each)
[green]4. Combine & deduplicate[/green] - Merge up to 40 results, remove duplicates
[green]5. AI rerank[/green] - Cohere analyzes all results, returns top 10 most relevant

[bold]Natural Language Query Examples:[/bold]

[dim]You:[/dim] "my client fell in a store on a slippery floor, illinois, last 5 years"
[dim]→ Semantic Query:[/dim] "customer injured slip fall premises liability negligence duty care store retail"
[dim]→ Keyword Query:[/dim] '"premises liability" AND (slip OR fall) AND (store OR retail) AND negligence'
[dim]→ Jurisdiction:[/dim] Illinois state + federal courts (ill illappct ca7 ilnd ilcd ilsd)
[dim]→ Dates:[/dim] 2020-01-09 to 2026-01-09
[dim]→ Results:[/dim] Best 10 from combined 40 results

[dim]You:[/dim] "qualified immunity excessive force, 9th circuit, recent cases"
[dim]→ Enriched:[/dim] "qualified immunity excessive force brutality unreasonable seizure fourth amendment"
[dim]→ Jurisdiction:[/dim] Ninth Circuit (ca9)
[dim]→ Dates:[/dim] Last 2 years (default for "recent")
[dim]→ Search:[/dim] Semantic (conceptual query)

[dim]You:[/dim] "breach of contract AND damages, california state only, since 2020"
[dim]→ Enriched:[/dim] (kept as-is - you used Boolean operators)
[dim]→ Jurisdiction:[/dim] California state courts only (cal calctapp)
[dim]→ Dates:[/dim] 2020-01-01 onwards
[dim]→ Search:[/dim] Keyword (you used AND operator)

[bold]You Can Be Vague![/bold]
• [cyan]"recent"[/cyan] → Last 2 years
• [cyan]"last 5 years"[/cyan] → Calculated from today
• [cyan]"California"[/cyan] → All CA state + federal courts
• [cyan]"federal"[/cyan] → All federal circuits + SCOTUS

[bold]Commands:[/bold]
• [cyan]exit[/cyan] - Quit
• [cyan]new[/cyan] - Start fresh conversation
• [cyan]jurisdictions[/cyan] - Show court codes
    """
    console.print(Panel(
        help_text,
        title="[bold]Help[/bold]",
        border_style="blue"
    ))


def display_jurisdictions():
    """Display the list of available jurisdictions/court codes."""
    # Federal Appellate Courts Table
    fed_table = Table(
        title="[bold cyan]Federal Appellate Courts[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold"
    )
    fed_table.add_column("Code", style="yellow", width=10)
    fed_table.add_column("Court Name", style="white")
    
    for code, name in FEDERAL_APPELLATE_COURTS.items():
        fed_table.add_row(code, name)
    
    console.print(fed_table)
    console.print()
    
    # Popular Jurisdictions
    pop_text = format_popular_jurisdictions()
    console.print(Panel(
        f"[bold]Popular Jurisdictions by Region:[/bold]{pop_text}",
        title="[bold green]Quick Reference[/bold green]",
        border_style="green"
    ))
    console.print()
    
    # State Supreme Courts Table
    state_table = Table(
        title="[bold magenta]State Supreme Courts[/bold magenta]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold"
    )
    state_table.add_column("Code", style="yellow", width=12)
    state_table.add_column("Court Name", style="white")
    
    for code, name in list(STATE_SUPREME_COURTS.items())[:20]:  # First 20 for brevity
        state_table.add_row(code, name)
    
    state_table.add_row("...", "[dim]Use specific code for other states[/dim]")
    
    console.print(state_table)
    console.print("\n[dim]Tip: You can combine multiple court codes with spaces, e.g., 'ca9 cal calctapp'[/dim]\n")


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text for plain display."""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def display_opinion_text(opinion_data: dict, case_name: str):
    """Display fetched opinion text."""
    if not opinion_data.get("success"):
        display_error(f"Could not fetch opinion: {opinion_data.get('error', 'Unknown error')}")
        return
    
    text = opinion_data.get("text", "")
    is_html = opinion_data.get("is_html", False)
    
    # Strip HTML if present
    if is_html:
        text = strip_html_tags(text)
    
    # Truncate for display (first ~3000 chars for prototype)
    if len(text) > 3000:
        text = text[:3000] + "\n\n[dim]... (text truncated for display)[/dim]"
    
    console.print(Panel(
        text,
        title=f"[bold cyan]Opinion Text: {case_name}[/bold cyan]",
        subtitle=f"[dim]Source field: {opinion_data.get('text_field', 'unknown')} | Opinion ID: {opinion_data.get('opinion_id')}[/dim]",
        border_style="cyan",
        padding=(1, 2)
    ))


def prompt_analyze_opinion(results: list, courtlistener_client: CourtListenerClient, 
                           llm_client=None, original_query: str = "") -> bool:
    """
    Prompt user to analyze an opinion after search results.
    
    Args:
        results: List of search results
        courtlistener_client: API client
        llm_client: Optional LLM client for AI analysis
        original_query: The user's original search query
        
    Returns:
        True if user wants to continue, False to exit analysis loop
    """
    if not results:
        return False
    
    console.print()
    console.print(Panel(
        "[bold]Would you like me to analyze any of these opinions?[/bold]\n\n"
        "[dim]Enter a number (1-{}) to analyze that opinion, or press Enter to skip.[/dim]\n"
        "[dim](Prototype: Can only analyze one opinion at a time)[/dim]".format(len(results)),
        title="[bold cyan]Yevrah[/bold cyan]",
        border_style="cyan"
    ))
    
    choice = input("> ").strip()
    
    if not choice:
        return False
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(results):
            console.print("[yellow]Invalid selection. Please enter a number between 1 and {}.[/yellow]".format(len(results)))
            return True  # Let them try again
        
        selected = results[idx]
        cluster_id = selected.get("cluster_id")
        case_name = selected.get("case_name", "Unknown Case")
        
        if not cluster_id:
            display_error("This case doesn't have a cluster ID. Cannot fetch opinion text.")
            return True
        
        console.print(f"\n[dim]Fetching full opinion text for: {case_name}...[/dim]")
        
        # Fetch the opinion text
        opinion_data = courtlistener_client.get_opinion_text_by_cluster(cluster_id)
        
        if not opinion_data.get("success"):
            display_error(f"Could not fetch opinion: {opinion_data.get('error', 'Unknown error')}")
            return True
        
        # If we have an LLM, use it to analyze
        if llm_client and llm_client.client:
            console.print("[dim]Analyzing opinion...[/dim]\n")
            
            # Prepare text for analysis (truncate if too long for context)
            text = opinion_data.get("text", "")
            is_html = opinion_data.get("is_html", False)
            if is_html:
                text = strip_html_tags(text)
            
            # Truncate for LLM context (first ~8000 chars)
            if len(text) > 8000:
                text = text[:8000] + "\n\n[TRUNCATED]"
            
            # Ask LLM to analyze
            analysis_prompt = f"""I've retrieved the full text of "{case_name}" based on the user's search for "{original_query}".

Please analyze this opinion and:
1. Summarize the key facts and holding (2-3 sentences)
2. Explain how this case is relevant to the user's query about "{original_query}"
3. Note any important legal principles or precedents established
4. Suggest how this case might be useful (or not useful) for the user's research

OPINION TEXT:
{text}"""
            
            # Add to chat and get response
            llm_client.chat_history.append({
                "role": "user",
                "content": analysis_prompt
            })
            
            try:
                from groq import Groq
                response = llm_client.client.chat.completions.create(
                    model=llm_client.model,
                    messages=llm_client.chat_history,
                    max_tokens=1500,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                llm_client.chat_history.append({
                    "role": "assistant",
                    "content": analysis
                })

                display_assistant_message(analysis)

            except Exception as e:
                display_error(f"Error during analysis: {str(e)}")
                # Fall back to just showing the text
                display_opinion_text(opinion_data, case_name)
        else:
            # No LLM - just display the text
            display_opinion_text(opinion_data, case_name)

        # Prompt user for next action after analysis
        console.print()
        console.print(Panel(
            "[bold]What would you like to do next?[/bold]\n\n"
            f"[dim]• Enter another number (1-{len(results)}) to analyze a different case[/dim]\n"
            "[dim]• Press Enter to continue with a new search[/dim]",
            title="[bold cyan]Yevrah[/bold cyan]",
            border_style="cyan"
        ))

        return True  # Allow another selection
        
    except ValueError:
        console.print("[yellow]Please enter a valid number.[/yellow]")
        return True


def run_fallback_mode(courtlistener_client: CourtListenerClient):
    """Run in fallback mode without LLM (guided search with validation)."""
    from tools import map_jurisdiction_to_codes, parse_date_input, extract_search_query
    
    console.print(Panel(
        "[yellow]Running in direct mode[/yellow] (no Groq API key)\n\n"
        "[dim]I'll guide you through building your search query step by step.[/dim]\n"
        "[dim]For full AI-assisted research, add GROQ_API_KEY to your .env file.[/dim]",
        title="[bold cyan]Yevrah[/bold cyan] [dim]- Direct Mode[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    while True:
        try:
            # Step 1: Get search topic
            console.print("\n[bold cyan]What are you researching?[/bold cyan]")
            console.print("[dim]Describe the legal issue (e.g., 'slip and fall', 'breach of contract'):[/dim]")
            search_topic = input("> ").strip()
            
            if search_topic.lower() in ['exit', 'quit', 'q']:
                console.print("\n[bold]Goodbye! Good luck with your research.[/bold]")
                break
            
            if search_topic.lower() == 'help':
                display_help()
                continue
            
            if search_topic.lower() in ['jurisdictions', 'courts']:
                display_jurisdictions()
                continue
            
            if not search_topic:
                continue
            
            # Step 2: Get jurisdiction
            console.print("\n[bold cyan]Which jurisdiction?[/bold cyan]")
            console.print("[dim]Examples: California, Ninth Circuit, ca9, federal (or press Enter for all):[/dim]")
            jurisdiction_input = input("> ").strip()
            
            # Validate jurisdiction
            jurisdiction_result = map_jurisdiction_to_codes(jurisdiction_input)
            if jurisdiction_result["court_codes"]:
                console.print(f"[dim]→ Courts: {jurisdiction_result['description']}[/dim]")
            
            # Step 3: Get date range
            console.print("\n[bold cyan]Date range?[/bold cyan]")
            console.print("[dim]Examples: last 3 years, 2020 to 2023, since 2020 (or press Enter for all time):[/dim]")
            date_input = input("> ").strip()
            
            # Validate date
            date_result = parse_date_input(date_input)
            if date_result["filed_after"] or date_result["filed_before"]:
                console.print(f"[dim]→ Period: {date_result['description']}[/dim]")
            
            # Step 4: Choose search type
            console.print("\n[bold cyan]Search type?[/bold cyan]")
            console.print("  [yellow]1[/yellow] - Keyword [dim](precise terms, Boolean operators)[/dim]")
            console.print("  [green]2[/green] - Semantic [dim](natural language, similar facts)[/dim]")
            choice = input("> ").strip()
            
            use_semantic = choice == "2"
            search_type = "Semantic" if use_semantic else "Keyword"
            
            # Step 5: Extract and clean up the search query
            extraction = extract_search_query(search_topic, use_keyword=not use_semantic)
            cleaned_query = extraction["query"]
            
            # Show what we extracted
            if cleaned_query != search_topic.lower().strip():
                console.print(f"\n[dim]→ Extracted query: \"{cleaned_query}\"[/dim]")
                if extraction["transformations"]:
                    for t in extraction["transformations"][:2]:  # Show first 2 transformations
                        console.print(f"[dim]  • {t}[/dim]")
            
            console.print(f"\n[dim]Searching ({search_type})...[/dim]")
            
            # Execute search with CLEANED query - limit to MAX_RESULTS
            results_data = courtlistener_client.search(
                query=cleaned_query,
                search_type="semantic" if use_semantic else "keyword",
                court=jurisdiction_result["court_codes"],
                filed_after=date_result["filed_after"],
                filed_before=date_result["filed_before"],
                page_size=MAX_RESULTS
            )
            
            # Display API URL
            if "_api_url" in results_data:
                display_api_url(results_data["_api_url"])
            
            # Check for errors
            if results_data.get("error"):
                display_error(f"API Error: {results_data['error']}")
                continue
            
            # Get results (limit to MAX_RESULTS)
            formatted_results = results_data.get("results", [])[:MAX_RESULTS]
            display_results(formatted_results, search_topic, search_type)
            
            # Prompt to analyze an opinion
            while formatted_results:
                if not prompt_analyze_opinion(formatted_results, courtlistener_client, 
                                             original_query=search_topic):
                    break
            
        except KeyboardInterrupt:
            console.print("\n\n[bold]Goodbye![/bold]")
            break
        except Exception as e:
            display_error(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Main application loop with conversational flow."""
    display_welcome()
    
    # Initialize CourtListener client
    try:
        courtlistener_client = CourtListenerClient()
    except ValueError as e:
        display_error(f"Configuration error: {str(e)}\nPlease add COURTLISTENER_API_KEY to .env")
        sys.exit(1)
    except Exception as e:
        display_error(f"Failed to initialize CourtListener client: {str(e)}")
        sys.exit(1)
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Check if running in fallback mode (no Groq API key)
    if llm_client.client is None:
        run_fallback_mode(courtlistener_client)
        return
    
    # Get and display welcome message from assistant
    console.print("[dim]Initializing assistant...[/dim]\n")
    welcome_response = llm_client.get_welcome_message()
    display_assistant_message(welcome_response)
    
    # Track last search results for analysis
    last_results = []
    last_query = ""
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            console.print("\n[bold cyan]You:[/bold cyan]", end=" ")
            user_input = input().strip()
            
            # Check for commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("\n[bold]Goodbye! Good luck with your research![/bold]")
                break
            
            if user_input.lower() in ['new', 'reset']:
                llm_client.reset_chat()
                last_results = []
                last_query = ""
                console.print("\n[dim]Starting new conversation...[/dim]\n")
                welcome_response = llm_client.get_welcome_message()
                display_assistant_message(welcome_response)
                continue
            
            if user_input.lower() == 'help':
                display_help()
                continue
            
            if user_input.lower() in ['jurisdictions', 'courts']:
                display_jurisdictions()
                continue
            
            if not user_input:
                continue
            
            # Check if user wants to analyze a result from last search
            if last_results and user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(last_results):
                    selected = last_results[idx]
                    cluster_id = selected.get("cluster_id")
                    case_name = selected.get("case_name", "Unknown Case")
                    
                    if cluster_id:
                        console.print(f"\n[dim]Fetching and analyzing: {case_name}...[/dim]")
                        
                        opinion_data = courtlistener_client.get_opinion_text_by_cluster(cluster_id)
                        
                        if opinion_data.get("success"):
                            # Prepare text for LLM
                            text = opinion_data.get("text", "")
                            if opinion_data.get("is_html"):
                                text = strip_html_tags(text)
                            if len(text) > 8000:
                                text = text[:8000] + "\n\n[TRUNCATED]"
                            
                            # Ask LLM to analyze THIS SINGLE CASE ONLY (no search capability)
                            analysis_prompt = f"""Analyze ONLY this single opinion: "{case_name}"

The user's original research query was: "{last_query}"

Provide a focused analysis of THIS CASE ONLY:
1. **Summary** - Key facts and holding (2-3 sentences)
2. **Relevance** - How this specific case relates to "{last_query}"
3. **Key Legal Principles** - Important precedents or rules established in this case
4. **Research Utility** - How useful is this particular case for the user's research?

Do NOT search for additional cases. Focus only on analyzing the opinion text below.

OPINION TEXT:
{text}"""
                            
                            # Use direct LLM call without search capability
                            console.print("[dim]Analyzing...[/dim]")
                            analysis = llm_client.analyze_opinion(analysis_prompt)
                            if analysis:
                                display_assistant_message(analysis)

                            # Prompt user for next action after analysis
                            console.print()
                            console.print(Panel(
                                "[bold]What would you like to do next?[/bold]\n\n"
                                f"[dim]• Enter another number (1-{len(last_results)}) to analyze a different case[/dim]\n"
                                "[dim]• Type 'new' to start a new search[/dim]",
                                title="[bold cyan]Yevrah[/bold cyan]",
                                border_style="cyan"
                            ))
                            # Don't fall through - explicitly continue to next iteration
                            continue
                        else:
                            display_error(f"Could not fetch opinion: {opinion_data.get('error', 'Unknown error')}")
                    else:
                        display_error("This case doesn't have a cluster ID.")
                    continue
                else:
                    console.print(f"[yellow]Please enter a number between 1 and {len(last_results)}.[/yellow]")
                    continue
            
            # Process the message through the LLM
            console.print("\n[dim]Thinking...[/dim]")
            
            result = llm_client.chat(user_input, courtlistener_client)
            
            # If a search was performed, show the search info
            if result["tool_called"] and result["search_results"]:
                # Pass metadata for dual search URL display
                metadata = result["search_results"].get("_metadata", {})
                display_search_info(
                    search_type=result["search_type"],
                    tool_args=result["tool_args"],
                    api_url=result["api_url"],
                    metadata=metadata
                )
                
                # Display the results (limited to MAX_RESULTS)
                if result["formatted_results"]:
                    formatted_results = result["formatted_results"][:MAX_RESULTS]
                    display_results(
                        formatted_results,
                        result["tool_args"].get("query", ""),
                        result["search_type"]
                    )
                    
                    # Store for potential analysis
                    last_results = formatted_results
                    last_query = result["tool_args"].get("query", user_input)
                    
                elif "error" in result["search_results"]:
                    display_error(f"Search error: {result['search_results']['error']}")
            
            # Display the assistant's response/explanation
            if result["response"]:
                display_assistant_message(result["response"])
            
            # If we have results, prompt for analysis
            if last_results and result["tool_called"]:
                console.print()
                console.print("[dim]Enter a number (1-{}) to analyze that opinion in detail, or continue chatting.[/dim]".format(len(last_results)))
            
        except KeyboardInterrupt:
            console.print("\n\n[bold]Goodbye! Good luck with your research![/bold]")
            break
        except Exception as e:
            display_error(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
