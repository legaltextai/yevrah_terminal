"""
Result formatting utilities using Rich library for beautiful terminal output.
"""
from typing import List, Dict, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box
from datetime import datetime

from jurisdictions import get_court_name, ALL_COURTS

console = Console()


def format_date(date_str: Optional[str]) -> str:
    """Format a date string for display."""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d")
    except:
        return date_str


def format_citation(citation: List) -> str:
    """Format citation list into readable string."""
    if not citation or not isinstance(citation, list):
        return "N/A"
    return ", ".join(str(c) for c in citation if c)


def display_assistant_message(message: str):
    """Display Yevrah's message in a styled panel."""
    console.print()
    console.print(Panel(
        Markdown(message),
        title="[bold cyan]Yevrah[/bold cyan]",
        border_style="cyan",
        padding=(1, 2)
    ))


def display_search_info(search_type: str, tool_args: Dict[str, Any], api_url: str, metadata: Dict[str, Any] = None):
    """
    Display information about the search being performed.

    Args:
        search_type: "Keyword", "Semantic", or "Dual (Keyword + Semantic)"
        tool_args: Arguments passed to the search tool
        api_url: The API URL used (for single searches)
        metadata: Optional metadata containing dual search API URLs
    """
    is_dual = "Dual" in search_type or search_type == "both"
    search_color = "magenta" if is_dual else ("green" if search_type == "Semantic" else "yellow")

    info_parts = [f"[bold {search_color}]{search_type} Search[/bold {search_color}]"]

    if tool_args:
        # Handle dual search queries
        if is_dual or tool_args.get("keyword_query"):
            semantic_query = tool_args.get("query", "")
            keyword_query = tool_args.get("keyword_query", "")

            if semantic_query:
                info_parts.append(f"\n[bold green]Semantic Query:[/bold green] \"{semantic_query}\"")
            if keyword_query:
                info_parts.append(f"\n[bold yellow]Keyword Query:[/bold yellow] \"{keyword_query}\"")
        else:
            # Single search query
            extracted_query = tool_args.get("extracted_query", tool_args.get("query", ""))
            if extracted_query:
                info_parts.append(f"\n[bold]Query:[/bold] \"{extracted_query}\"")

        # Handle jurisdiction - prefer expanded value from metadata, fall back to tool_args
        jurisdiction = None
        if metadata:
            jurisdiction = metadata.get("court", "")
        if not jurisdiction:
            jurisdiction = tool_args.get("jurisdiction", tool_args.get("court", ""))
        if jurisdiction:
            info_parts.append(f"\n[dim]Jurisdiction:[/dim] {jurisdiction}")
        else:
            info_parts.append(f"\n[dim]Jurisdiction:[/dim] All courts")

        # Handle date range - could be from new or old format
        date_range = tool_args.get("date_range", "")
        if date_range:
            info_parts.append(f"\n[dim]Date Range:[/dim] {date_range}")
        else:
            # Check for filed_after/filed_before (new format) or date_filed_min/date_filed_max (old format)
            date_min = tool_args.get("filed_after", tool_args.get("date_filed_min", ""))
            date_max = tool_args.get("filed_before", tool_args.get("date_filed_max", ""))
            if date_min or date_max:
                info_parts.append(f"\n[dim]Date Range:[/dim] {date_min or 'any'} to {date_max or 'present'}")
            else:
                info_parts.append(f"\n[dim]Date Range:[/dim] All time")

        # Show reasoning if available (from new search_case_law tool)
        reasoning = tool_args.get("reasoning", "")
        if reasoning:
            info_parts.append(f"\n\n[dim italic]Reasoning: {reasoning}[/dim italic]")

    console.print()
    console.print(Panel(
        "".join(info_parts),
        title="[bold]Search Parameters[/bold]",
        border_style=search_color
    ))

    # Display API URLs
    if is_dual and metadata:
        # For dual search, show both API URLs
        keyword_url = metadata.get("keyword_api_url", "")
        semantic_url = metadata.get("semantic_api_url", "")

        if keyword_url or semantic_url:
            display_dual_api_urls(keyword_url, semantic_url)
    elif api_url and not is_dual:
        # For single search, show single API URL
        display_api_url(api_url)


def display_api_url(api_url: str):
    """Display the generated API URL (Cmd+click to open)."""
    console.print(Panel(
        f"[underline cyan]{api_url}[/underline cyan]",
        title="[bold]API Request URL[/bold] [dim](Cmd+click to open)[/dim]",
        border_style="cyan"
    ))


def display_dual_api_urls(keyword_url: str, semantic_url: str):
    """Display both API URLs for dual search (Cmd+click to open)."""
    url_parts = []

    if keyword_url:
        url_parts.append(f"[bold yellow]Keyword Search:[/bold yellow]\n[underline cyan]{keyword_url}[/underline cyan]")

    if semantic_url:
        url_parts.append(f"[bold green]Semantic Search:[/bold green]\n[underline cyan]{semantic_url}[/underline cyan]")

    if url_parts:
        console.print(Panel(
            "\n\n".join(url_parts),
            title="[bold magenta]API Request URLs[/bold magenta] [dim](Cmd+click to open)[/dim]",
            border_style="magenta",
            padding=(1, 2)
        ))


def display_search_query(query: str, search_type: str = "Keyword"):
    """Display the search query that was used."""
    search_type_label = "[bold green]Semantic[/bold green]" if search_type == "Semantic" else "[bold yellow]Keyword[/bold yellow]"
    console.print(f"\n[bold cyan]Search Query:[/bold cyan] [dim]{query}[/dim] ({search_type_label}[dim])[/dim]\n")


def prompt_search_type() -> bool:
    """
    Prompt user to choose between keyword and semantic search.
    
    Returns:
        True if semantic search, False if keyword search
    """
    while True:
        console.print("\n[bold yellow]Choose search type:[/bold yellow]")
        console.print("  [dim]1[/dim] - [bold]Keyword[/bold] (BM25, exact word matching, supports operators)")
        console.print("  [dim]2[/dim] - [bold]Semantic[/bold] (natural language understanding)")
        console.print()
        choice = input("[bold cyan]Enter choice (1 or 2):[/bold cyan] ").strip()
        
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            console.print("[red]Invalid choice. Please enter 1 or 2.[/red]")


def display_results(results: List[Dict], search_query: str, search_type: str = "Keyword"):
    """
    Display formatted search results in the terminal.
    
    Args:
        results: List of formatted result dictionaries
        search_query: The search query that was used
        search_type: Type of search used ("Keyword" or "Semantic")
    """
    if not results:
        console.print(Panel(
            "[yellow]No results found for your query.[/yellow]\n"
            "Try adjusting your search terms or using different keywords.",
            title="No Results",
            border_style="yellow"
        ))
        return
    
    # Create a summary table
    table = Table(
        title=f"[bold green]Found {len(results)} Case(s)[/bold green]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("#", style="dim", width=3)
    table.add_column("Case Name", style="cyan", no_wrap=False, width=40)
    table.add_column("Court", style="yellow", width=20)
    table.add_column("Date Filed", style="green", width=12)
    table.add_column("Citation", style="blue", width=18)
    
    for idx, result in enumerate(results, 1):
        case_name = result.get("case_name", "N/A")
        if len(case_name) > 45:
            case_name = case_name[:42] + "..."
        court = result.get("court", "N/A")
        if len(court) > 22:
            court = court[:19] + "..."
        date_filed = format_date(result.get("date_filed"))
        citation = format_citation(result.get("citation"))
        if len(citation) > 20:
            citation = citation[:17] + "..."
        
        table.add_row(str(idx), case_name, court, date_filed, citation)
    
    console.print()
    console.print(table)
    
    # Display detailed cards for each result
    console.print("\n[bold]Case Details:[/bold]\n")
    
    for idx, result in enumerate(results, 1):
        display_case_card(idx, result)


def format_snippet(snippet: str, max_length: int = None) -> str:
    """
    Format a snippet for display, converting HTML highlight marks to Rich markup.

    Args:
        snippet: Raw snippet text (may contain <mark> tags)
        max_length: Optional maximum length before truncation (None = no truncation)

    Returns:
        Formatted snippet string with Rich markup
    """
    if not snippet:
        return ""

    # Clean up whitespace
    snippet = " ".join(snippet.split())

    # Convert <mark> tags to Rich bold/yellow for highlighting
    snippet = snippet.replace("<mark>", "[bold yellow]")
    snippet = snippet.replace("</mark>", "[/bold yellow]")

    # Truncate if needed and max_length is specified
    if max_length and len(snippet) > max_length:
        # Find a good breaking point
        truncate_at = snippet.rfind(" ", 0, max_length)
        if truncate_at == -1:
            truncate_at = max_length
        snippet = snippet[:truncate_at] + "..."

    return snippet


def display_case_card(idx: int, result: Dict):
    """Display a single case result as a card."""
    case_name = result.get("case_name", "N/A")
    case_name_full = result.get("case_name_full", "")
    court = result.get("court", "N/A")
    court_citation = result.get("court_citation_string", "")
    date_filed = format_date(result.get("date_filed"))
    date_argued = format_date(result.get("date_argued"))
    citation = format_citation(result.get("citation"))
    docket_number = result.get("docket_number", "N/A")
    absolute_url = result.get("absolute_url", "")
    url = result.get("url", "")
    cite_count = result.get("cite_count", 0)
    judge = result.get("judge", "")
    status = result.get("status", "")
    download_url = result.get("download_url", "")
    cluster_id = result.get("cluster_id")

    # Search source (keyword or semantic)
    search_source = result.get("_search_source", "")

    # Relevance scores
    score_bm25 = result.get("score_bm25")
    score_semantic = result.get("score_semantic")
    
    # Get snippet - try top-level first, then opinions[0]
    snippet = result.get("snippet", "")
    if not snippet and result.get("opinions"):
        snippet = result["opinions"][0].get("snippet", "") if result["opinions"] else ""
    
    # Build content
    lines = []
    lines.append(f"[bold cyan]Case Name:[/bold cyan] {case_name}")
    if case_name_full and case_name_full != case_name:
        lines.append(f"[dim]Full Name: {case_name_full}[/dim]")
    
    court_line = f"[bold yellow]Court:[/bold yellow] {court}"
    if court_citation:
        court_line += f" [dim]({court_citation})[/dim]"
    lines.append(court_line)
    
    if date_filed != "N/A":
        lines.append(f"[bold green]Date Filed:[/bold green] {date_filed}")
    if date_argued and date_argued != "N/A":
        lines.append(f"[dim]Date Argued: {date_argued}[/dim]")
    
    if citation != "N/A":
        lines.append(f"[bold blue]Citation:[/bold blue] {citation}")
    
    if docket_number != "N/A":
        lines.append(f"[dim]Docket Number:[/dim] {docket_number}")
    
    if judge:
        lines.append(f"[bold magenta]Judge(s):[/bold magenta] {judge}")
    
    if cite_count > 0:
        lines.append(f"[dim]Cited by {cite_count} case(s)[/dim]")
    
    if status:
        lines.append(f"[dim]Status: {status}[/dim]")
    
    # Relevance scores
    score_parts = []
    if score_bm25 is not None:
        score_parts.append(f"BM25: {score_bm25:.2f}")
    if score_semantic is not None:
        score_parts.append(f"Semantic: {score_semantic:.2f}")
    if score_parts:
        lines.append(f"[bold magenta]Relevance:[/bold magenta] {' | '.join(score_parts)}")
    
    # Cluster ID
    if cluster_id:
        lines.append(f"[dim]Cluster ID: {cluster_id}[/dim]")
    
    # URL - prefer the full url, fall back to constructing from absolute_url
    # Note: Using underline style makes URLs auto-detected by most terminals (Cmd+click)
    view_url = url if url else (f"https://www.courtlistener.com{absolute_url}" if absolute_url else "")
    if view_url:
        lines.append(f"[bold blue]View:[/bold blue] [underline cyan]{view_url}[/underline cyan]")
    
    if download_url:
        lines.append(f"[bold green]PDF:[/bold green] [underline cyan]{download_url}[/underline cyan]")
    
    # Snippet with highlighting (display full snippet)
    if snippet:
        formatted_snippet = format_snippet(snippet)
        lines.append(f"\n[bold]Snippet:[/bold]\n[dim]{formatted_snippet}[/dim]")
    
    content = "\n".join(lines)

    # Build title with source tag
    title_parts = [f"[bold]#{idx}[/bold]"]
    if search_source == "keyword":
        title_parts.append("[bold yellow on black] KEYWORD [/bold yellow on black]")
    elif search_source == "semantic":
        title_parts.append("[bold green on black] SEMANTIC [/bold green on black]")

    title = " ".join(title_parts)

    console.print(Panel(
        content,
        title=title,
        border_style="blue",
        padding=(0, 1)
    ))
    console.print()


def display_error(message: str):
    """Display an error message."""
    console.print(Panel(
        f"[red]{message}[/red]",
        title="[bold red]Error[/bold red]",
        border_style="red"
    ))


def display_info(message: str):
    """Display an info message."""
    console.print(Panel(
        f"[blue]{message}[/blue]",
        title="Info",
        border_style="blue"
    ))


def display_welcome():
    """Display welcome message (legacy - kept for compatibility)."""
    welcome_text = """
[bold cyan]AI-Enabled Research Assistant[/bold cyan]

I'm here to help you find relevant case law using the CourtListener database.

[dim]Type 'exit', 'quit', or 'q' to exit.[/dim]
    """
    console.print(Panel(
        welcome_text,
        title="Welcome",
        border_style="cyan"
    ))
    console.print()


def display_non_case_law_message(explanation: str):
    """Display message when query is not a case law query."""
    console.print(Panel(
        f"[yellow]{explanation}[/yellow]",
        title="Not a Case Law Query",
        border_style="yellow"
    ))
