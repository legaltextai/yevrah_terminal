"""
Tool definitions for Groq function calling.
Defines tools for keyword and semantic search on CourtListener API.

Reference: 
- https://console.groq.com/docs/tool-use/overview
- https://console.groq.com/docs/structured-outputs
"""
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from jurisdictions import ALL_COURTS, search_courts


# =============================================================================
# QUERY EXTRACTION - Clean up natural language to search terms
# =============================================================================

# Phrases to remove from user queries
NOISE_PHRASES = [
    # Request phrases
    r'\b(find\s+me|i\s+need|looking\s+for|search\s+for|get\s+me|show\s+me|can\s+you\s+find)\b',
    r'\b(i\'m\s+looking\s+for|i\s+am\s+looking\s+for|i\s+want)\b',
    r'\b(please\s+find|please\s+search|please\s+get)\b',
    # Case references
    r'\bcases?\s+(about|regarding|involving|related\s+to|on|for|where|in\s+which)\b',
    r'\b(case\s+law|case\s+laws|legal\s+cases?)\s+(about|regarding|on|for)\b',
    r'\bcases?\s+from\b',
    r'\bcases?\s+in\b',
    r'\bcases?\b',  # Remove standalone "case" or "cases"
    r'\bcourts?\b',  # Remove standalone "court" or "courts"
    r'\bin\s+a\b',  # Remove "in a"
    # Location phrases WITH context (we extract these separately via jurisdiction filter)
    r'\bin\s+(california|ca|cal|new\s+york|ny|texas|tx|florida|fl|illinois|il)\b',
    r'\bfrom\s+(the\s+)?(9th|ninth|first|second|third|fourth|fifth|sixth|seventh|eighth|tenth|eleventh|dc|federal)\s*circuit\b',
    r'\bfrom\s+(california|ca|cal|new\s+york|ny|texas|tx|florida|fl)\b',
    # Time phrases (we extract these separately via date filter)
    r'\bfrom\s+(the\s+)?(last|past)\s+\d+\s+years?\b',
    r'\b(last|past)\s+\d+\s+years?\b',
    r'\bsince\s+\d{4}\b',
    r'\bbefore\s+\d{4}\b',
    r'\b\d{4}\s*to\s*\d{4}\b',
]

# Standalone jurisdiction names to remove (handled separately by jurisdiction filter)
JURISDICTION_WORDS = [
    r'\b(california|ca|cal)\b',
    r'\b(new\s+york|ny|nyc)\b',
    r'\b(texas|tx)\b',
    r'\b(florida|fl|fla)\b',
    r'\b(illinois|il)\b',
    r'\b(georgia|ga)\b',
    r'\b(ohio|oh)\b',
    r'\b(michigan|mi)\b',
    r'\b(pennsylvania|pa)\b',
    r'\b(federal|state)\b',
    r'\b(ninth|9th|first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|seventh|7th|eighth|8th|tenth|10th|eleventh|11th|dc)\s*circuit\b',
    r'\b(scotus|supreme\s+court)\b',
]

# Common legal term expansions for keyword search
LEGAL_TERM_EXPANSIONS = {
    'slip and fall': 'slip AND fall',
    'slip & fall': 'slip AND fall',
    'breach of contract': 'breach AND contract',
    'breach of fiduciary duty': '"breach of fiduciary duty"',
    'premises liability': '"premises liability"',
    'wrongful termination': '"wrongful termination"',
    'employment discrimination': '"employment discrimination"',
    'personal injury': '"personal injury"',
    'medical malpractice': '"medical malpractice"',
    'product liability': '"product liability"',
    'intellectual property': '"intellectual property"',
    'trade secret': '"trade secret"',
    'non compete': '"non-compete" OR "noncompete"',
    'due process': '"due process"',
    'equal protection': '"equal protection"',
    'first amendment': '"first amendment"',
    'fourth amendment': '"fourth amendment"',
    'qualified immunity': '"qualified immunity"',
}


def extract_search_query(raw_input: str, use_keyword: bool = False) -> Dict[str, Any]:
    """
    Extract the core legal search terms from natural language input.
    
    Args:
        raw_input: The user's natural language query
        use_keyword: If True, optimize for keyword search with Boolean operators
        
    Returns:
        Dictionary with:
        - query: The cleaned/extracted search query
        - original: The original input
        - transformations: List of transformations applied
    """
    if not raw_input:
        return {
            "query": "",
            "original": raw_input,
            "transformations": []
        }
    
    transformations = []
    query = raw_input.lower().strip()
    original = query
    
    # Step 1: Remove noise phrases
    for pattern in NOISE_PHRASES:
        before = query
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        if query != before:
            transformations.append(f"Removed noise phrase")
    
    # Step 1b: Remove standalone jurisdiction words (handled separately)
    for pattern in JURISDICTION_WORDS:
        before = query
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        if query != before:
            transformations.append(f"Removed jurisdiction reference")
    
    # Step 2: Clean up extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Step 3: Fix common typos EARLY (before keyword expansion)
    typo_fixes = {
        r'\bfal\b': 'fall',
        r'\bneglgence\b': 'negligence',
        r'\bliablity\b': 'liability',
        r'\bcontarct\b': 'contract',
        r'\bemploy\b': 'employment',
    }
    for typo, fix in typo_fixes.items():
        before = query
        query = re.sub(typo, fix, query, flags=re.IGNORECASE)
        if query != before:
            transformations.append(f"Fixed typo: {typo} → {fix}")
    
    # Step 4: Remove leading/trailing common words
    query = re.sub(r'^(the|a|an|some|any)\s+', '', query)
    query = re.sub(r'\s+(the|a|an)$', '', query)
    
    # Step 5: For keyword search, apply Boolean transformations
    if use_keyword:
        expansion_applied = False
        
        # Check for known legal term patterns
        query_lower = query.lower()
        for term, expansion in LEGAL_TERM_EXPANSIONS.items():
            if term in query_lower:
                # Replace the term with its expansion
                query = re.sub(re.escape(term), expansion, query, flags=re.IGNORECASE)
                transformations.append(f"Applied legal term expansion for '{term}'")
                expansion_applied = True
                break
        
        # Only add AND between words if NO expansion was applied and no operators exist
        if not expansion_applied:
            words = query.split()
            if len(words) >= 2 and 'AND' not in query.upper() and 'OR' not in query.upper() and '"' not in query:
                # Check if it looks like a phrase vs separate concepts
                if len(words) <= 4:
                    # Short phrase - use AND
                    query = ' AND '.join(words)
                    transformations.append(f"Added AND operators")
    
    # Final cleanup
    query = query.strip()
    
    # If query became empty, fall back to original (minus obvious noise)
    if not query or len(query) < 3:
        query = re.sub(r'\b(find|me|need|looking|for|search|cases?|about)\b', '', original, flags=re.IGNORECASE)
        query = re.sub(r'\s+', ' ', query).strip()
        transformations.append("Query too short, using cleaned original")
    
    return {
        "query": query,
        "original": raw_input,
        "transformations": transformations
    }

# =============================================================================
# JURISDICTION MAPPING - Natural Language to Court Codes
# =============================================================================

# =============================================================================
# STATE COURT MAPPING - Comprehensive mappings for all US jurisdictions
# Format: state supreme + state appellate + federal circuit + federal districts
# =============================================================================

STATE_COURT_MAPPING = {
    # ----- ALABAMA (11th Circuit) -----
    "alabama": "ala alactapp alacrimapp ca11 almd alnd alsd",
    "alabama state": "ala alactapp alacrimapp",
    "alabama federal": "ca11 almd alnd alsd",
    "al": "ala alactapp alacrimapp ca11 almd alnd alsd",
    
    # ----- ALASKA (9th Circuit) -----
    "alaska": "alaska alaskactapp ca9 akd",
    "alaska state": "alaska alaskactapp",
    "alaska federal": "ca9 akd",
    "ak": "alaska alaskactapp ca9 akd",
    
    # ----- ARIZONA (9th Circuit) -----
    "arizona": "ariz arizctapp ca9 azd",
    "arizona state": "ariz arizctapp",
    "arizona federal": "ca9 azd",
    "az": "ariz arizctapp ca9 azd",
    
    # ----- ARKANSAS (8th Circuit) -----
    "arkansas": "ark arkctapp ca8 ared arwd",
    "arkansas state": "ark arkctapp",
    "arkansas federal": "ca8 ared arwd",
    "ar": "ark arkctapp ca8 ared arwd",
    
    # ----- CALIFORNIA (9th Circuit) -----
    "california": "cal calctapp calappdeptsuper ca9 cacd caed cand casd californiad cacb caeb canb casb calag",
    "california state": "cal calctapp calappdeptsuper",
    "california federal": "ca9 cacd caed cand casd californiad cacb caeb canb casb",
    "ca": "cal calctapp calappdeptsuper ca9 cacd caed cand casd californiad cacb caeb canb casb calag",
    "calif": "cal calctapp calappdeptsuper ca9 cacd caed cand casd californiad cacb caeb canb casb calag",
    
    # ----- COLORADO (10th Circuit) -----
    "colorado": "colo coloctapp ca10 cod",
    "colorado state": "colo coloctapp",
    "colorado federal": "ca10 cod",
    "co": "colo coloctapp ca10 cod",
    
    # ----- CONNECTICUT (2nd Circuit) -----
    "connecticut": "conn connappct ca2 ctd",
    "connecticut state": "conn connappct",
    "connecticut federal": "ca2 ctd",
    "ct": "conn connappct ca2 ctd",
    
    # ----- DELAWARE (3rd Circuit) -----
    "delaware": "del ca3 ded",
    "delaware state": "del",
    "delaware federal": "ca3 ded",
    "de": "del ca3 ded",
    
    # ----- DISTRICT OF COLUMBIA (DC Circuit) -----
    "district of columbia": "dc cadc dcd",
    "washington dc": "dc cadc dcd",
    "d.c.": "dc cadc dcd",
    "dc": "dc cadc dcd",
    
    # ----- FLORIDA (11th Circuit) -----
    "florida": "fla flactapp ca11 flmd flnd flsd flmb flnb flsb flaag",
    "florida state": "fla flactapp",
    "florida federal": "ca11 flmd flnd flsd flmb flnb flsb",
    "fl": "fla flactapp ca11 flmd flnd flsd flmb flnb flsb flaag",
    "fla": "fla flactapp ca11 flmd flnd flsd flmb flnb flsb flaag",
    
    # ----- GEORGIA (11th Circuit) -----
    "georgia": "ga gactapp ca11 gamd gand gasd",
    "georgia state": "ga gactapp",
    "georgia federal": "ca11 gamd gand gasd",
    "ga": "ga gactapp ca11 gamd gand gasd",
    
    # ----- HAWAII (9th Circuit) -----
    "hawaii": "haw hawctapp ca9 hid",
    "hawaii state": "haw hawctapp",
    "hawaii federal": "ca9 hid",
    "hi": "haw hawctapp ca9 hid",
    
    # ----- IDAHO (9th Circuit) -----
    "idaho": "idaho idahoctapp ca9 idd",
    "idaho state": "idaho idahoctapp",
    "idaho federal": "ca9 idd",
    "id": "idaho idahoctapp ca9 idd",
    
    # ----- ILLINOIS (7th Circuit) -----
    "illinois": "ill illappct ca7 ilcd ilnd ilsd",
    "illinois state": "ill illappct",
    "illinois federal": "ca7 ilcd ilnd ilsd",
    "il": "ill illappct ca7 ilcd ilnd ilsd",
    
    # ----- INDIANA (7th Circuit) -----
    "indiana": "ind indctapp ca7 innd insd",
    "indiana state": "ind indctapp",
    "indiana federal": "ca7 innd insd",
    "in": "ind indctapp ca7 innd insd",
    
    # ----- IOWA (8th Circuit) -----
    "iowa": "iowa iowactapp ca8 iand iasd",
    "iowa state": "iowa iowactapp",
    "iowa federal": "ca8 iand iasd",
    "ia": "iowa iowactapp ca8 iand iasd",
    
    # ----- KANSAS (10th Circuit) -----
    "kansas": "kan kanctapp ca10 ksd",
    "kansas state": "kan kanctapp",
    "kansas federal": "ca10 ksd",
    "ks": "kan kanctapp ca10 ksd",
    
    # ----- KENTUCKY (6th Circuit) -----
    "kentucky": "ky kyctapp ca6 kyed kywd",
    "kentucky state": "ky kyctapp",
    "kentucky federal": "ca6 kyed kywd",
    "ky": "ky kyctapp ca6 kyed kywd",
    
    # ----- LOUISIANA (5th Circuit) -----
    "louisiana": "la lactapp ca5 laed lamd lawd",
    "louisiana state": "la lactapp",
    "louisiana federal": "ca5 laed lamd lawd",
    "la": "la lactapp ca5 laed lamd lawd",
    
    # ----- MAINE (1st Circuit) -----
    "maine": "me ca1 med",
    "maine state": "me",
    "maine federal": "ca1 med",
    "me": "me ca1 med",
    
    # ----- MARYLAND (4th Circuit) -----
    "maryland": "md mdctspecapp ca4 mdd",
    "maryland state": "md mdctspecapp",
    "maryland federal": "ca4 mdd",
    "md": "md mdctspecapp ca4 mdd",
    
    # ----- MASSACHUSETTS (1st Circuit) -----
    "massachusetts": "mass massappct ca1 mad",
    "massachusetts state": "mass massappct",
    "massachusetts federal": "ca1 mad",
    "ma": "mass massappct ca1 mad",
    
    # ----- MICHIGAN (6th Circuit) -----
    "michigan": "mich michctapp ca6 mied miwd",
    "michigan state": "mich michctapp",
    "michigan federal": "ca6 mied miwd",
    "mi": "mich michctapp ca6 mied miwd",
    
    # ----- MINNESOTA (8th Circuit) -----
    "minnesota": "minn minnctapp ca8 mnd",
    "minnesota state": "minn minnctapp",
    "minnesota federal": "ca8 mnd",
    "mn": "minn minnctapp ca8 mnd",
    
    # ----- MISSISSIPPI (5th Circuit) -----
    "mississippi": "miss missctapp ca5 msnd mssd",
    "mississippi state": "miss missctapp",
    "mississippi federal": "ca5 msnd mssd",
    "ms": "miss missctapp ca5 msnd mssd",
    
    # ----- MISSOURI (8th Circuit) -----
    "missouri": "mo moctapp ca8 moed mowd",
    "missouri state": "mo moctapp",
    "missouri federal": "ca8 moed mowd",
    "mo": "mo moctapp ca8 moed mowd",
    
    # ----- MONTANA (9th Circuit) -----
    "montana": "mont ca9 mtd",
    "montana state": "mont",
    "montana federal": "ca9 mtd",
    "mt": "mont ca9 mtd",
    
    # ----- NEBRASKA (8th Circuit) -----
    "nebraska": "neb nebctapp ca8 ned",
    "nebraska state": "neb nebctapp",
    "nebraska federal": "ca8 ned",
    "ne": "neb nebctapp ca8 ned",
    
    # ----- NEVADA (9th Circuit) -----
    "nevada": "nev nevapp ca9 nvd",
    "nevada state": "nev nevapp",
    "nevada federal": "ca9 nvd",
    "nv": "nev nevapp ca9 nvd",
    
    # ----- NEW HAMPSHIRE (1st Circuit) -----
    "new hampshire": "nh ca1 nhd",
    "new hampshire state": "nh",
    "new hampshire federal": "ca1 nhd",
    "nh": "nh ca1 nhd",
    
    # ----- NEW JERSEY (3rd Circuit) -----
    "new jersey": "nj njsuperctappdiv ca3 njd",
    "new jersey state": "nj njsuperctappdiv",
    "new jersey federal": "ca3 njd",
    "nj": "nj njsuperctappdiv ca3 njd",
    
    # ----- NEW MEXICO (10th Circuit) -----
    "new mexico": "nm nmctapp ca10 nmd",
    "new mexico state": "nm nmctapp",
    "new mexico federal": "ca10 nmd",
    "nm": "nm nmctapp ca10 nmd",
    
    # ----- NEW YORK (2nd Circuit) -----
    "new york": "ny nyappdiv nyappterm ca2 nyed nynd nysd nywd",
    "new york state": "ny nyappdiv nyappterm",
    "new york federal": "ca2 nyed nynd nysd nywd",
    "ny": "ny nyappdiv nyappterm ca2 nyed nynd nysd nywd",
    "nyc": "ny nyappdiv ca2 nysd nyed",
    
    # ----- NORTH CAROLINA (4th Circuit) -----
    "north carolina": "nc ncctapp ca4 nced ncmd ncwd",
    "north carolina state": "nc ncctapp",
    "north carolina federal": "ca4 nced ncmd ncwd",
    "nc": "nc ncctapp ca4 nced ncmd ncwd",
    
    # ----- NORTH DAKOTA (8th Circuit) -----
    "north dakota": "nd ndctapp ca8 ndd",
    "north dakota state": "nd ndctapp",
    "north dakota federal": "ca8 ndd",
    "nd": "nd ndctapp ca8 ndd",
    
    # ----- OHIO (6th Circuit) -----
    "ohio": "ohio ohioctapp ca6 ohnd ohsd",
    "ohio state": "ohio ohioctapp",
    "ohio federal": "ca6 ohnd ohsd",
    "oh": "ohio ohioctapp ca6 ohnd ohsd",
    
    # ----- OKLAHOMA (10th Circuit) -----
    "oklahoma": "okla oklacivapp oklacrimapp ca10 oked oknd okwd",
    "oklahoma state": "okla oklacivapp oklacrimapp",
    "oklahoma federal": "ca10 oked oknd okwd",
    "ok": "okla oklacivapp oklacrimapp ca10 oked oknd okwd",
    
    # ----- OREGON (9th Circuit) -----
    "oregon": "or orctapp ca9 ord",
    "oregon state": "or orctapp",
    "oregon federal": "ca9 ord",
    "or": "or orctapp ca9 ord",
    
    # ----- PENNSYLVANIA (3rd Circuit) -----
    "pennsylvania": "pa pasuperct pacommwct ca3 paed pamd pawd",
    "pennsylvania state": "pa pasuperct pacommwct",
    "pennsylvania federal": "ca3 paed pamd pawd",
    "penn": "pa pasuperct pacommwct ca3 paed pamd pawd",
    "pa": "pa pasuperct pacommwct ca3 paed pamd pawd",
    
    # ----- RHODE ISLAND (1st Circuit) -----
    "rhode island": "ri ca1 rid",
    "rhode island state": "ri",
    "rhode island federal": "ca1 rid",
    "ri": "ri ca1 rid",
    
    # ----- SOUTH CAROLINA (4th Circuit) -----
    "south carolina": "sc scctapp ca4 scd",
    "south carolina state": "sc scctapp",
    "south carolina federal": "ca4 scd",
    "sc": "sc scctapp ca4 scd",
    
    # ----- SOUTH DAKOTA (8th Circuit) -----
    "south dakota": "sd ca8 sdd",
    "south dakota state": "sd",
    "south dakota federal": "ca8 sdd",
    "sd": "sd ca8 sdd",
    
    # ----- TENNESSEE (6th Circuit) -----
    "tennessee": "tenn tennctapp tenncrimapp ca6 tned tnmd tnwd",
    "tennessee state": "tenn tennctapp tenncrimapp",
    "tennessee federal": "ca6 tned tnmd tnwd",
    "tn": "tenn tennctapp tenncrimapp ca6 tned tnmd tnwd",
    
    # ----- TEXAS (5th Circuit) -----
    "texas": "tex texcrimapp texapp ca5 txed txnd txsd txwd",
    "texas state": "tex texcrimapp texapp",
    "texas federal": "ca5 txed txnd txsd txwd",
    "tx": "tex texcrimapp texapp ca5 txed txnd txsd txwd",
    
    # ----- UTAH (10th Circuit) -----
    "utah": "utah utahctapp ca10 utd",
    "utah state": "utah utahctapp",
    "utah federal": "ca10 utd",
    "ut": "utah utahctapp ca10 utd",
    
    # ----- VERMONT (2nd Circuit) -----
    "vermont": "vt ca2 vtd",
    "vermont state": "vt",
    "vermont federal": "ca2 vtd",
    "vt": "vt ca2 vtd",
    
    # ----- VIRGINIA (4th Circuit) -----
    "virginia": "va vactapp ca4 vaed vawd",
    "virginia state": "va vactapp",
    "virginia federal": "ca4 vaed vawd",
    "va": "va vactapp ca4 vaed vawd",
    
    # ----- WASHINGTON (9th Circuit) -----
    "washington": "wash washctapp ca9 waed wawd",
    "washington state": "wash washctapp",
    "washington federal": "ca9 waed wawd",
    "wa": "wash washctapp ca9 waed wawd",
    
    # ----- WEST VIRGINIA (4th Circuit) -----
    "west virginia": "wva ca4 wvnd wvsd",
    "west virginia state": "wva",
    "west virginia federal": "ca4 wvnd wvsd",
    "wv": "wva ca4 wvnd wvsd",
    
    # ----- WISCONSIN (7th Circuit) -----
    "wisconsin": "wis wisctapp ca7 wied wiwd",
    "wisconsin state": "wis wisctapp",
    "wisconsin federal": "ca7 wied wiwd",
    "wi": "wis wisctapp ca7 wied wiwd",
    
    # ----- WYOMING (10th Circuit) -----
    "wyoming": "wyo ca10 wyd",
    "wyoming state": "wyo",
    "wyoming federal": "ca10 wyd",
    "wy": "wyo ca10 wyd",
    
    # ==========================================================================
    # FEDERAL CIRCUIT COURTS
    # ==========================================================================
    "first circuit": "ca1",
    "1st circuit": "ca1",
    "second circuit": "ca2",
    "2nd circuit": "ca2",
    "third circuit": "ca3",
    "3rd circuit": "ca3",
    "fourth circuit": "ca4",
    "4th circuit": "ca4",
    "fifth circuit": "ca5",
    "5th circuit": "ca5",
    "sixth circuit": "ca6",
    "6th circuit": "ca6",
    "seventh circuit": "ca7",
    "7th circuit": "ca7",
    "eighth circuit": "ca8",
    "8th circuit": "ca8",
    "ninth circuit": "ca9",
    "9th circuit": "ca9",
    "tenth circuit": "ca10",
    "10th circuit": "ca10",
    "eleventh circuit": "ca11",
    "11th circuit": "ca11",
    "dc circuit": "cadc",
    "d.c. circuit": "cadc",
    "federal circuit": "cafc",
    
    # ==========================================================================
    # SUPREME COURT
    # ==========================================================================
    "supreme court": "scotus",
    "us supreme court": "scotus",
    "united states supreme court": "scotus",
    "scotus": "scotus",
    
    # ==========================================================================
    # ALL FEDERAL COURTS
    # ==========================================================================
    "federal": "scotus ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc",
    "all federal": "scotus ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc",
    "federal appellate": "ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc",
}


def map_jurisdiction_to_codes(jurisdiction_input: str) -> Dict[str, Any]:
    """
    Map a natural language jurisdiction input to validated court codes.

    Priority order:
    1. STATE_COURT_MAPPING (expands state names to all relevant courts)
    2. Direct court codes (if user provides specific codes like "ca9 cal")
    3. Courts-DB fuzzy matching
    4. Local fuzzy search on court names

    Supports natural language variations:
    - "california" / "California" / "CALIFORNIA"
    - "california state" / "California State Courts"
    - "texas federal" / "Texas Federal Courts"
    - "state of california" / "federal courts in texas"
    - "new york state courts" / "NY state"
    """
    if not jurisdiction_input:
        return {
            "valid": True,
            "court_codes": "",
            "description": "All courts (no jurisdiction filter)",
            "suggestion": ""
        }

    jurisdiction_lower = jurisdiction_input.lower().strip()

    # Normalize common variations
    # Remove noise words and rearrange common patterns
    normalized = jurisdiction_lower

    # Handle "state of X" → "X" or "X state"
    state_of_match = re.search(r'state\s+of\s+(\w+(?:\s+\w+)?)', normalized)
    if state_of_match:
        state_name = state_of_match.group(1)
        normalized = state_name  # "state of california" → "california"

    # Handle "federal courts in X" → "X federal"
    federal_in_match = re.search(r'federal\s+courts?\s+in\s+(\w+(?:\s+\w+)?)', normalized)
    if federal_in_match:
        state_name = federal_in_match.group(1)
        normalized = f"{state_name} federal"  # "federal courts in texas" → "texas federal"

    # Remove remaining noise words: "courts", "court", "the", "in", "of"
    normalized = re.sub(r'\b(courts?)\b', '', normalized)  # Remove "court" or "courts"
    normalized = re.sub(r'\b(the|in|of)\b', '', normalized)  # Remove "the", "in", "of"
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Collapse multiple spaces

    # FIRST: Check STATE_COURT_MAPPING for natural language state names
    # Try both original and normalized versions
    # This handles cases like "ohio", "iowa", "idaho" which are also court codes
    # but users likely mean the full state jurisdiction
    if jurisdiction_lower in STATE_COURT_MAPPING:
        codes = STATE_COURT_MAPPING[jurisdiction_lower]
        code_list = codes.split()
        descriptions = [ALL_COURTS.get(code, code) for code in code_list[:5]]
        desc = ", ".join(descriptions)
        if len(code_list) > 5:
            desc += f" and {len(code_list) - 5} more"
        return {
            "valid": True,
            "court_codes": codes,
            "description": desc,
            "suggestion": ""
        }

    # Try normalized version (e.g., "California State Courts" → "california state")
    if normalized and normalized != jurisdiction_lower and normalized in STATE_COURT_MAPPING:
        codes = STATE_COURT_MAPPING[normalized]
        code_list = codes.split()
        descriptions = [ALL_COURTS.get(code, code) for code in code_list[:5]]
        desc = ", ".join(descriptions)
        if len(code_list) > 5:
            desc += f" and {len(code_list) - 5} more"
        return {
            "valid": True,
            "court_codes": codes,
            "description": desc,
            "suggestion": f"Normalized from '{jurisdiction_input}' to '{normalized}'"
        }
    
    # SECOND: Check if it's already valid court codes (for specific codes like "ca9 cal")
    input_codes = jurisdiction_lower.split()
    all_valid_codes = True
    for code in input_codes:
        if code not in ALL_COURTS:
            all_valid_codes = False
            break
    
    if all_valid_codes:
        descriptions = [ALL_COURTS.get(code, code) for code in input_codes]
        return {
            "valid": True,
            "court_codes": jurisdiction_lower,
            "description": ", ".join(descriptions),
            "suggestion": ""
        }
    
    # THIRD: Try local fuzzy matching
    matches = search_courts(jurisdiction_lower)
    if matches:
        codes = " ".join(list(matches.keys())[:10])
        descriptions = list(matches.values())[:5]
        desc = ", ".join(descriptions)
        if len(matches) > 5:
            desc += f" and {len(matches) - 5} more"
        return {
            "valid": True,
            "court_codes": codes,
            "description": desc,
            "suggestion": f"Found {len(matches)} matching courts"
        }

    return {
        "valid": False,
        "court_codes": "",
        "description": "",
        "suggestion": f"Could not recognize '{jurisdiction_input}'. Try state names (e.g., 'California'), "
                      f"circuit names (e.g., 'Ninth Circuit'), or court codes (e.g., 'ca9', 'cal')."
    }


def parse_date_input(date_input: str) -> Dict[str, Any]:
    """
    Parse various date input formats into MM/DD/YYYY format.
    """
    if not date_input:
        return {
            "valid": True,
            "filed_after": "",
            "filed_before": "",
            "description": "All time (no date filter)"
        }
    
    date_lower = date_input.lower().strip()
    today = datetime.now()
    
    # Pattern: "last X years" or "past X years"
    last_years_match = re.search(r'(?:last|past)\s+(\d+)\s+years?', date_lower)
    if last_years_match:
        years = int(last_years_match.group(1))
        start_date = today - timedelta(days=years * 365)
        return {
            "valid": True,
            "filed_after": start_date.strftime("%m/%d/%Y"),
            "filed_before": today.strftime("%m/%d/%Y"),
            "description": f"Last {years} year(s)"
        }
    
    # Pattern: "YYYY to YYYY" or "YYYY-YYYY"
    range_match = re.search(r'(\d{4})\s*(?:to|-)\s*(\d{4})', date_lower)
    if range_match:
        start_year = range_match.group(1)
        end_year = range_match.group(2)
        return {
            "valid": True,
            "filed_after": f"01/01/{start_year}",
            "filed_before": f"12/31/{end_year}",
            "description": f"{start_year} to {end_year}"
        }
    
    # Pattern: "since YYYY" or "after YYYY"
    since_match = re.search(r'(?:since|after|from)\s+(\d{4})', date_lower)
    if since_match:
        year = since_match.group(1)
        return {
            "valid": True,
            "filed_after": f"01/01/{year}",
            "filed_before": today.strftime("%m/%d/%Y"),
            "description": f"Since {year}"
        }
    
    # Pattern: "before YYYY" or "until YYYY"
    before_match = re.search(r'(?:before|until)\s+(\d{4})$', date_lower)
    if before_match:
        year = before_match.group(1)
        return {
            "valid": True,
            "filed_after": "",
            "filed_before": f"12/31/{year}",
            "description": f"Before {year}"
        }
    
    # Pattern: Just a year "2020"
    year_match = re.match(r'^(\d{4})$', date_lower)
    if year_match:
        year = year_match.group(1)
        return {
            "valid": True,
            "filed_after": f"01/01/{year}",
            "filed_before": f"12/31/{year}",
            "description": f"Year {year}"
        }
    
    # Pattern: MM/DD/YYYY
    mmddyyyy_match = re.match(r'^(\d{2}/\d{2}/\d{4})$', date_lower)
    if mmddyyyy_match:
        return {
            "valid": True,
            "filed_after": date_lower,
            "filed_before": "",
            "description": f"After {date_lower}"
        }
    
    # Pattern: YYYY-MM-DD
    isodate_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', date_lower)
    if isodate_match:
        year, month, day = isodate_match.groups()
        formatted = f"{month}/{day}/{year}"
        return {
            "valid": True,
            "filed_after": formatted,
            "filed_before": "",
            "description": f"After {formatted}"
        }
    
    return {
        "valid": False,
        "filed_after": "",
        "filed_before": "",
        "description": f"Could not parse date: '{date_input}'. Try 'last 3 years', '2020 to 2023', or 'since 2020'."
    }


# =============================================================================
# TOOL DEFINITIONS FOR GROQ FUNCTION CALLING
# Reference: 
# - https://console.groq.com/docs/tool-use/overview
# - https://www.courtlistener.com/help/search-operators/
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_case_law",
            "description": """Search CourtListener's database of 8+ million legal opinions.

DUAL SEARCH (RECOMMENDED): Runs both keyword and semantic searches in parallel, fetches 20 results from each, combines and reranks all 40 results, returns top 5 most relevant.

KEYWORD SEARCH: Uses exact term matching with Boolean operators (AND, OR, NOT), wildcards (*), proximity (~N), fielded searches (field:value), and range queries ([X TO Y]).

SEMANTIC SEARCH: Uses natural language understanding to find cases with similar meaning/facts. No operators supported - just descriptive phrases.

Returns case metadata, relevance scores, and text snippets highlighting matching content.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """The semantic search query (enriched with legal concepts and synonyms).

If search_type is "both", this will be used for the semantic search.
If search_type is "semantic", this is the only query used.
If search_type is "keyword", this is the only query used (should contain Boolean operators).

FOR SEMANTIC (or BOTH): Use enriched natural descriptive language:
- Add legal terminology and concepts
- Include synonyms and related terms
- Describe factual scenario
- NO operators (they won't work in semantic search)

FOR KEYWORD: Use Boolean operators and CourtListener syntax:
- Boolean: AND, OR, NOT, - (negation)
- Phrases: "exact phrase"
- Wildcards: negligen* (ending), gr*mm*r (middle)
- Proximity: "breach duty"~10
- Fuzzy: immigrant~2
- Fields: court_id:ca9, dateFiled:[2020-01-01 TO *], citeCount:[50 TO *]
- Groups: (slip OR trip OR fall) AND premises

Examples:
- Semantic/Both: 'customer injured slip fall premises liability negligence duty care store retail'
- Keyword: '"premises liability" AND (slip OR fall) AND negligence'"""
                    },
                    "keyword_query": {
                        "type": "string",
                        "description": """REQUIRED if search_type is "both". The keyword-optimized query with Boolean operators.

This should be different from the main query and optimized for keyword search:
- Use Boolean operators (AND, OR, NOT)
- Include exact phrases in quotes
- Use wildcards for variations
- Add proximity searches for concepts

Example: '"premises liability" AND (slip OR fall) AND (grocery OR store OR retail) AND negligence'"""
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["both", "keyword", "semantic"],
                        "description": """Which search strategy to use:

BOTH (RECOMMENDED): Runs keyword and semantic searches in parallel, combines 40 results, reranks with AI, returns top 5. Provides comprehensive coverage.

KEYWORD: Best for precise legal terms, Boolean logic, specific courts/dates, highly-cited cases, exact phrases. Supports all operators.

SEMANTIC: Best for factual scenarios, plain language descriptions, finding similar cases, exploratory research. No operators.

Use "both" unless user explicitly requests only keyword or only semantic."""
                    },
                    "court": {
                        "type": "string",
                        "description": """Filter by court(s). Use CourtListener court_id values, space-separated for multiple.

Federal Appeals: ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc
Supreme Court: scotus
District Courts: cacd caed cand casd nysd nyed txsd txed (etc.)
State Courts: cal calctapp ny nyappdiv tex texapp (etc.)

Examples:
- '9th Circuit only': 'ca9'
- 'All California federal': 'cacd caed cand casd'
- 'California state': 'cal calctapp'

Leave empty for all courts."""
                    },
                    "filed_after": {
                        "type": "string",
                        "description": "Filter to cases filed on or after this date. Format: MM/DD/YYYY or YYYY-MM-DD. Example: '01/01/2020'"
                    },
                    "filed_before": {
                        "type": "string",
                        "description": "Filter to cases filed on or before this date. Format: MM/DD/YYYY or YYYY-MM-DD. Example: '12/31/2024'"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["published", "unpublished", "all"],
                        "description": "Precedential status filter. 'published' = binding precedent (default), 'unpublished' = non-precedential, 'all' = both."
                    },
                    "order_by": {
                        "type": "string",
                        "enum": ["score desc", "dateFiled desc", "dateFiled asc", "citeCount desc"],
                        "description": """Sort order for results:
- 'score desc': By relevance (default, recommended)
- 'dateFiled desc': Newest first
- 'dateFiled asc': Oldest first  
- 'citeCount desc': Most cited first"""
                    },
                    "cited_gt": {
                        "type": "integer",
                        "description": "Minimum citation count. Only return cases cited more than this number. Example: 50 for influential cases."
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results to return (1-20). Default 5."
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor from previous response's 'next_cursor' field. Use to fetch subsequent pages of results."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of your search strategy: why you chose keyword vs semantic, how you formulated the query, what you expect to find."
                    }
                },
                "required": ["query", "search_type"]
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTION FUNCTIONS
# =============================================================================

def format_tool_call_args(tool_call) -> Dict[str, Any]:
    """Extract and parse arguments from a tool call."""
    try:
        return json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, AttributeError):
        return {}


def convert_boolean_to_natural_language(query: str) -> str:
    """
    Convert a Boolean query to natural language for semantic search.

    CourtListener supports Boolean operators: AND (or &), OR, NOT (or %)
    Also handles: quoted phrases, parentheses grouping, wildcards (*)

    Examples:
        "employment AND discrimination" → "employment discrimination"
        "slip AND fall OR premises liability" → "slip fall or premises liability"
        "negligence NOT intentional" → "negligence but not intentional"
        "asylum & immigration" → "asylum immigration"
        "border % patrol" → "border but not patrol"

    Args:
        query: Query string potentially containing Boolean operators

    Returns:
        Natural language version of the query
    """
    import re

    # Check if query contains Boolean operators (AND, OR, NOT, &, %)
    has_boolean = bool(re.search(r'(\b(AND|OR|NOT)\b|[&%])', query, re.IGNORECASE))

    if not has_boolean:
        return query

    # Convert Boolean operators to natural language
    result = query

    # Handle & (alternative AND operator) - just remove it
    result = re.sub(r'\s*&\s*', ' ', result)

    # Handle AND - just remove it (terms adjacent in natural language)
    result = re.sub(r'\s+AND\s+', ' ', result, flags=re.IGNORECASE)

    # Handle % and NOT together - convert to "but not"
    # First normalize % to NOT, then handle all NOT consistently
    result = re.sub(r'\s*%\s*', ' NOT ', result)
    result = re.sub(r'\s+NOT\s+', ' but not ', result, flags=re.IGNORECASE)

    # Handle OR - convert to lowercase "or"
    result = re.sub(r'\s+OR\s+', ' or ', result, flags=re.IGNORECASE)

    # Clean up parentheses - remove them for natural language
    # but preserve the words inside
    result = re.sub(r'[()]', ' ', result)

    # Clean up any double spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def execute_search_case_law(arguments: Dict[str, Any], courtlistener_client) -> Dict[str, Any]:
    """
    Execute the search_case_law tool with dual search support.

    Supports three modes:
    - "both": Runs keyword and semantic searches in parallel, combines 40 results, reranks to top 5
    - "keyword": Single keyword search, fetches 20, reranks to top 5
    - "semantic": Single semantic search, fetches 20, reranks to top 5

    Supports both new schema (query, keyword_query, search_type, court, filed_after, etc.)
    and legacy schema (extracted_query, jurisdiction, date_range).
    """
    import concurrent.futures

    # Extract parameters - support both new and legacy schemas
    original_query = arguments.get("query", arguments.get("extracted_query", ""))
    keyword_query = arguments.get("keyword_query", "")
    requested_search_type = arguments.get("search_type", "semantic")  # Default from LLM is usually semantic
    reasoning = arguments.get("reasoning", "")

    # Detect if user provided Boolean operators and convert for semantic search
    import re
    has_boolean_operators = bool(re.search(r'(\b(AND|OR|NOT)\b|[&%])', original_query, re.IGNORECASE))

    if has_boolean_operators:
        # User provided Boolean query
        # Keep original for keyword search, convert to natural language for semantic
        keyword_query = original_query if not keyword_query else keyword_query
        query = convert_boolean_to_natural_language(original_query)
    else:
        # Normal natural language query
        query = original_query

    # FORCE DUAL SEARCH BY DEFAULT - unless user explicitly requested single search type
    # Check if this looks like an explicit single-search request (very rare)
    explicit_single_search = (
        requested_search_type in ["keyword", "semantic"] and
        "only" in reasoning.lower()  # User said "keyword only" or "semantic only"
    )

    if explicit_single_search:
        # Respect explicit user request for single search
        search_type = requested_search_type
    else:
        # ALWAYS use dual search (this is the default behavior)
        search_type = "both"

        # Auto-generate keyword query if not provided
        if not keyword_query:
            import re
            # Extract key legal terms (words with 4+ letters, excluding common words)
            words = re.findall(r'\b[a-z]{4,}\b', query.lower())
            common_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'what', 'when', 'where', 'which', 'while', 'about', 'after', 'before', 'because', 'through', 'during', 'between'}
            key_terms = [w for w in words if w not in common_words][:7]  # Take top 7 terms
            if key_terms:
                keyword_query = " AND ".join(key_terms)
            else:
                keyword_query = query  # Fallback to semantic query

    # Court filter - new schema uses 'court' directly, legacy uses 'jurisdiction'
    court = arguments.get("court", "")
    if not court and arguments.get("jurisdiction"):
        # Legacy: map natural language jurisdiction to court codes
        jurisdiction_result = map_jurisdiction_to_codes(arguments.get("jurisdiction", ""))
        court = jurisdiction_result.get("court_codes", "")

    # Date filters - new schema uses filed_after/filed_before, legacy uses date_range
    filed_after = arguments.get("filed_after", "")
    filed_before = arguments.get("filed_before", "")
    if not filed_after and not filed_before and arguments.get("date_range"):
        # Legacy: parse natural language date range
        date_result = parse_date_input(arguments.get("date_range", ""))
        filed_after = date_result.get("filed_after", "")
        filed_before = date_result.get("filed_before", "")

    # Additional parameters from new schema
    status = arguments.get("status", "published")
    order_by = arguments.get("order_by", "score desc")
    cited_gt = arguments.get("cited_gt")
    page_size_per_search = 20  # Fetch 20 from each search
    keyword_top_n = 5  # Show top 5 keyword results (by BM25)
    semantic_top_n = 5  # Show top 5 semantic results (after Cohere reranking)
    cursor = arguments.get("cursor")

    # Build metadata for display and debugging
    metadata = {
        "semantic_query": query,
        "keyword_query": keyword_query,
        "search_type": search_type,
        "court": court,
        "filed_after": filed_after,
        "filed_before": filed_before,
        "status": status,
        "order_by": order_by,
        "cited_gt": cited_gt,
        "page_size_per_search": page_size_per_search,
        "keyword_top_n": keyword_top_n,
        "semantic_top_n": semantic_top_n,
        "reasoning": reasoning,
        "semantic_reranked": False,
        "dual_search": search_type == "both"
    }

    # Execute the search
    if courtlistener_client is None:
        return {
            "success": False,
            "error": "Search client not available",
            "results": [],
            "count": 0,
            "_metadata": metadata
        }

    try:
        final_results = []

        if search_type == "both":
            # Dual search: Run both keyword and semantic searches in parallel
            if not keyword_query:
                # Fall back to using query for both if keyword_query not provided
                keyword_query = query

            def run_keyword_search():
                return courtlistener_client.search(
                    query=keyword_query,
                    search_type="keyword",
                    court=court,
                    filed_after=filed_after,
                    filed_before=filed_before,
                    status=status,
                    order_by=order_by,
                    cited_gt=cited_gt,
                    page_size=page_size_per_search,
                    cursor=cursor
                )

            def run_semantic_search():
                return courtlistener_client.search(
                    query=query,
                    search_type="semantic",
                    court=court,
                    filed_after=filed_after,
                    filed_before=filed_before,
                    status=status,
                    order_by=order_by,
                    cited_gt=cited_gt,
                    page_size=page_size_per_search,
                    cursor=cursor
                )

            # Run both searches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                keyword_future = executor.submit(run_keyword_search)
                semantic_future = executor.submit(run_semantic_search)

                keyword_results = keyword_future.result()
                semantic_results = semantic_future.result()

            # Get results from both searches
            keyword_cases = keyword_results.get("results", [])
            semantic_cases = semantic_results.get("results", [])

            # Tag keyword results and take top 5 by BM25 score
            keyword_top_5 = []
            for case in keyword_cases[:keyword_top_n]:
                case["_search_source"] = "keyword"
                keyword_top_5.append(case)

            # Rerank semantic results with Cohere (if available)
            semantic_top_5 = []
            try:
                from reranker import CohereReranker
                reranker = CohereReranker()

                if reranker.is_available() and len(semantic_cases) > 0:
                    # Rerank semantic results only
                    reranked_semantic = reranker.rerank(
                        query=query,
                        documents=semantic_cases,
                        top_n=semantic_top_n,
                        return_documents=True
                    )
                    semantic_top_5 = reranked_semantic
                    metadata["semantic_reranked"] = True
                else:
                    # No reranker available, take top 5 from CourtListener's order
                    semantic_top_5 = semantic_cases[:semantic_top_n]
            except Exception as e:
                import logging
                logging.getLogger("tools").warning(f"Semantic reranking failed: {e}. Using top {semantic_top_n} from CourtListener.")
                semantic_top_5 = semantic_cases[:semantic_top_n]

            # Tag semantic results
            for case in semantic_top_5:
                case["_search_source"] = "semantic"

            # Combine: keyword results first, then semantic results
            final_results = keyword_top_5 + semantic_top_5

            metadata["keyword_results_count"] = len(keyword_cases)
            metadata["semantic_results_count"] = len(semantic_cases)
            metadata["keyword_shown"] = len(keyword_top_5)
            metadata["semantic_shown"] = len(semantic_top_5)
            metadata["keyword_api_url"] = keyword_results.get("_api_url", "")
            metadata["semantic_api_url"] = semantic_results.get("_api_url", "")

        else:
            # Single search (keyword or semantic)
            actual_search_type = "semantic" if search_type == "semantic" else "keyword"
            actual_query = query if search_type == "semantic" else (keyword_query if keyword_query else query)

            results = courtlistener_client.search(
                query=actual_query,
                search_type=actual_search_type,
                court=court,
                filed_after=filed_after,
                filed_before=filed_before,
                status=status,
                order_by=order_by,
                cited_gt=cited_gt,
                page_size=page_size_per_search,
                cursor=cursor
            )

            all_cases = results.get("results", [])

            # For semantic single search, rerank if available
            if search_type == "semantic" and len(all_cases) > semantic_top_n:
                try:
                    from reranker import CohereReranker
                    reranker = CohereReranker()

                    if reranker.is_available():
                        reranked = reranker.rerank(
                            query=query,
                            documents=all_cases,
                            top_n=semantic_top_n,
                            return_documents=True
                        )
                        final_results = reranked
                        metadata["semantic_reranked"] = True
                    else:
                        final_results = all_cases[:semantic_top_n]
                except Exception as e:
                    import logging
                    logging.getLogger("tools").warning(f"Semantic reranking failed: {e}")
                    final_results = all_cases[:semantic_top_n]
            else:
                # Keyword search or fewer results than needed - just take top results
                top_n = keyword_top_n if search_type == "keyword" else semantic_top_n
                final_results = all_cases[:top_n]

            # Tag results with source
            for case in final_results:
                case["_search_source"] = actual_search_type

        return {
            "success": True,
            "count": len(final_results),
            "results": final_results,
            "_metadata": metadata,
            "pagination": {"has_more": False},  # Dual search doesn't support pagination
            "_api_url": f"Dual search: keyword={keyword_query}, semantic={query}" if search_type == "both" else "",
            "_search_type": "Dual (Keyword + Semantic)" if search_type == "both" else ("Semantic" if search_type == "semantic" else "Keyword")
        }

    except Exception as e:
        # Handle any errors gracefully
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "results": [],
            "count": 0,
            "_metadata": metadata
        }


def execute_tool(tool_name: str, arguments: Dict[str, Any], courtlistener_client=None) -> Dict[str, Any]:
    """
    Execute a tool call and return results.
    """
    if tool_name == "search_case_law":
        return execute_search_case_law(arguments, courtlistener_client)
    
    # Legacy support for old tool names
    if tool_name in ["keyword_search", "semantic_search", "build_search_query"]:
        # Convert to new format
        query = arguments.get("query", arguments.get("legal_topic", ""))
        court = arguments.get("court", "")
        date_min = arguments.get("date_filed_min", "")
        date_max = arguments.get("date_filed_max", "")
        use_semantic = tool_name == "semantic_search"
        
        if courtlistener_client is None:
            return {"error": "Search client not available", "results": [], "count": 0}
        
        return courtlistener_client.search_filtered(
            query=query,
            court=court,
            date_filed_min=date_min,
            date_filed_max=date_max,
            semantic=use_semantic
        )
    
    return {"error": f"Unknown tool: {tool_name}", "results": [], "count": 0}


def format_results_for_llm(results_data: Dict, formatted_results: List[Dict] = None) -> str:
    """
    Format search results as a string for the LLM to explain.
    Includes pagination info and error handling details.
    """
    # Handle errors
    if results_data.get("error"):
        error_msg = results_data.get("error", "Unknown error")
        error_type = results_data.get("error_type", "")
        
        summary = f"Search error occurred: {error_msg}"
        if error_type:
            summary += f"\nError type: {error_type}"
        
        # Add helpful suggestions based on error type
        if "Invalid" in error_msg or "query" in error_msg.lower():
            summary += "\n\nSuggestion: Try simplifying the query or check for syntax errors in Boolean operators."
        elif "Rate" in error_msg:
            summary += "\n\nSuggestion: Wait a moment and try again, or narrow the search criteria."
        elif "Authentication" in error_msg:
            summary += "\n\nSuggestion: Check that the API token is valid."
        
        return summary
    
    count = results_data.get("count", 0)
    
    # Use formatted_results if provided, otherwise extract from results_data
    if formatted_results is None:
        formatted_results = results_data.get("results", [])
    
    # Include metadata about the search
    metadata = results_data.get("_metadata", {})
    meta = results_data.get("meta", {})
    
    summary_parts = []
    
    # Search info
    if metadata or meta:
        summary_parts.append("Search executed:")
        query = metadata.get("cleaned_query") or metadata.get("original_query") or meta.get("query", "N/A")
        summary_parts.append(f"- Query: \"{query}\"")
        search_type = metadata.get("search_type") or meta.get("search_type", "N/A")
        summary_parts.append(f"- Type: {search_type}")
        
        if metadata.get("court"):
            summary_parts.append(f"- Courts: {metadata.get('court')}")
        if metadata.get("filed_after") or metadata.get("filed_before"):
            date_range = f"{metadata.get('filed_after', 'any')} to {metadata.get('filed_before', 'present')}"
            summary_parts.append(f"- Date Range: {date_range}")
        if metadata.get("status") and metadata.get("status") != "published":
            summary_parts.append(f"- Status: {metadata.get('status')}")
        if metadata.get("cited_gt"):
            summary_parts.append(f"- Minimum Citations: {metadata.get('cited_gt')}")
        
        summary_parts.append("")
    
    # No results handling
    if not formatted_results:
        summary_parts.append("No results found for this search.")
        summary_parts.append("\nSuggestions:")
        summary_parts.append("- Try broader search terms")
        summary_parts.append("- Remove date or court restrictions")
        summary_parts.append("- Try semantic search for more conceptual matching")
        return "\n".join(summary_parts)
    
    # Results summary
    summary_parts.append(f"Found {count} total results. Here are the top {len(formatted_results)}:\n")
    
    for i, result in enumerate(formatted_results, 1):
        case_name = result.get("case_name", "Unknown")

        # Include citation next to case name
        citations = result.get("citation", [])
        if citations and isinstance(citations, list):
            citation_str = ', '.join(str(c) for c in citations)
            summary_parts.append(f"{i}. {case_name}, {citation_str}")
        else:
            summary_parts.append(f"{i}. {case_name}")

        summary_parts.append(f"   Court: {result.get('court', 'N/A')}")
        summary_parts.append(f"   Date Filed: {result.get('date_filed', 'N/A')}")
        
        # Cite count (authority indicator)
        cite_count = result.get("cite_count", 0)
        if cite_count > 0:
            summary_parts.append(f"   Times Cited: {cite_count}")
        
        # Status
        status = result.get("status")
        if status:
            summary_parts.append(f"   Status: {status}")
        
        # Snippet
        snippet = result.get("snippet", "")
        if not snippet and result.get("opinions"):
            snippet = result["opinions"][0].get("snippet", "") if result["opinions"] else ""
        if snippet:
            snippet_clean = " ".join(snippet.split())[:250]
            summary_parts.append(f"   Snippet: {snippet_clean}...")
        
        # URL
        url = result.get("url", "")
        if url:
            summary_parts.append(f"   URL: {url}")
        
        summary_parts.append("")
    
    # Pagination info
    pagination = results_data.get("pagination", {})
    if pagination.get("has_more"):
        summary_parts.append(f"[More results available - {count - len(formatted_results)} remaining. Use cursor to fetch next page.]")
    
    return "\n".join(summary_parts)
