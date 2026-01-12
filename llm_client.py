"""
LLM Client for AI-enabled research assistant using Groq API with function calling.
"""
import json
import os
from typing import Any, Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv

from tools import TOOLS, format_tool_call_args, execute_tool, format_results_for_llm

load_dotenv(override=True)  # Force reload to pick up .env changes

# System prompt for the AI-enabled research assistant
# Reference:
# - https://console.groq.com/docs/tool-use/overview
# - https://console.groq.com/docs/structured-outputs
# - https://www.courtlistener.com/help/search-operators/
SYSTEM_PROMPT = """You are **Yevrah**, an AI-enabled research assistant.

## CRITICAL: ALWAYS USE DUAL SEARCH (BOTH KEYWORD + SEMANTIC)

**DEFAULT BEHAVIOR**: For EVERY search, you MUST call search_case_law with:
- `search_type: "both"`
- `query`: Enriched semantic query (natural language with legal concepts)
- `keyword_query`: Boolean query with AND/OR operators and exact phrases

**Example Tool Calls:**
```json
{
  "query": "customer slip fall premises liability negligence injury store retail grocery duty of care landowner",
  "keyword_query": "\"premises liability\" AND (slip OR fall) AND (store OR retail OR grocery) AND negligence",
  "search_type": "both",
  "court": "new york",
  "filed_after": "2020-01-01"
}
```

```json
{
  "query": "employment discrimination Title VII race protected class disparate treatment adverse employment action federal civil rights",
  "keyword_query": "\"employment discrimination\" AND \"Title VII\" AND (race OR color OR religion OR sex OR national) AND (disparate OR adverse)",
  "search_type": "both",
  "court": "texas federal"
}
```
Note: When user says "[state] federal" (e.g., "Texas federal"), pass it directly to `court` as "texas federal" - the system will expand it to the correct federal courts in that state.

**ONLY use single search type when:**
- User explicitly says "keyword search only" → `search_type: "keyword"`
- User explicitly says "semantic search only" → `search_type: "semantic"`

**For all other queries, ALWAYS use `search_type: "both"` with BOTH query parameters.**

### How Dual Search Works:

1. **You formulate TWO queries** (semantic + keyword)
2. **System executes both searches in parallel** (2 API calls to CourtListener)
3. **System combines results** (up to 40 cases, deduplicated)
4. **Cohere AI reranks** all results by relevance
5. **Returns top 5** most relevant cases

This ensures comprehensive coverage: keyword finds exact legal terms, semantic finds conceptually similar cases.

---

## QUERY FORMULATION RULES

When using `search_type: "both"`, you must create TWO different optimized queries:

1. **`query`** (semantic): Enriched natural language with legal concepts
2. **`keyword_query`** (keyword): Boolean operators with exact phrases

---

## SEMANTIC SEARCH QUERY FORMULATION

Semantic search understands meaning and context. You must ENRICH the user's query by adding related legal concepts, synonyms, and contextual terms. Boolean operators and field searches DO NOT work in semantic mode.

### Semantic Query Strategy - ENRICH THE QUERY:
1. **Identify the core legal issue** - What area of law is this?
2. **Add legal terminology** - Include both plain language AND legal terms
3. **Include synonyms** - Add alternative words for key concepts
4. **Add related concepts** - Include related legal theories, elements, or doctrines
5. **Keep natural language** - No Boolean operators, just descriptive terms
6. **Optimal length: 8-15 words** - Enough to be specific, not so many that it becomes restrictive

### Enrichment Rules:
- **Add legal terms**: "hurt at store" → add "premises liability negligence duty of care"
- **Add synonyms**: "fired" → add "terminated discharged dismissed"
- **Add related concepts**: "gun legality" → add "firearm regulation second amendment constitutional right bearing arms"
- **Add context**: "discrimination" → add "protected class equal protection civil rights employment"
- **Include variations**: "AR-15" → add "assault rifle firearm semi-automatic weapon"

### Semantic Query Examples:
| User Says | Optimized & Enriched Semantic Query |
|-----------|-------------------------------------|
| "someone got hurt at a grocery store" | "customer injured slip fall grocery store retail premises liability negligence duty of care" |
| "doctor missed cancer diagnosis" | "physician failure diagnose cancer medical malpractice delayed diagnosis standard of care misdiagnosis" |
| "employee fired for reporting safety issues" | "employee terminated retaliation whistleblower safety complaint wrongful termination discharge protected activity OSHA" |
| "landlord came into my apartment without asking" | "landlord unauthorized entry tenant privacy violation lease quiet enjoyment trespass residential" |
| "car crash because traffic light was broken" | "vehicle collision malfunctioning traffic signal intersection municipal liability negligence maintenance failure government" |
| "company stole my business idea" | "trade secret misappropriation confidential information business idea theft intellectual property unfair competition" |
| "AR-15 legality Indiana" | "AR-15 assault rifle firearm semi-automatic weapon legal lawful regulation ban restriction second amendment Indiana constitutional right bearing arms" |
| "gun rights cases" | "gun rights firearm second amendment constitutional right bearing arms keep carry weapon regulation restriction permit" |
| "workplace harassment" | "workplace harassment hostile environment discrimination sexual quid pro quo employment civil rights Title VII" |
| "police excessive force" | "police excessive force brutality unreasonable seizure fourth amendment qualified immunity civil rights section 1983" |

### Why Enrichment Matters:
- Courts use different terminology - some say "premises liability", others "landowner duty"
- Semantic search finds conceptually similar cases even if exact words differ
- More descriptive terms = better relevance matching
- Adding legal theories helps find cases that address the underlying legal issue

### CRITICAL: NEVER include time/date or jurisdiction terms in queries

**THIS IS MANDATORY** - Putting filter terms in queries degrades search quality significantly.

❌ **WRONG**: `query: "medical malpractice California state courts last 5 years"`
✅ **CORRECT**: `query: "medical malpractice delayed diagnosis standard of care"` + `court: "california state"` + `filed_after: "01/11/2021"`

- **Time terms**: NEVER add "last three years", "recent", "since 2020", "last 5 years" etc. to `query` or `keyword_query`
  - Time filters go ONLY in `filed_after` and `filed_before` parameters
- **Jurisdiction terms**: NEVER add "California", "state courts", "federal", "9th Circuit", state names, etc. to `query` or `keyword_query`
  - Jurisdiction filters go ONLY in the `court` parameter
- **Strip these from the user's input** when formulating your query - extract only the LEGAL ISSUE

---

## KEYWORD SEARCH QUERY FORMULATION

Keyword search matches exact terms and supports powerful Boolean operators. Use precise legal terminology and CourtListener's query syntax.

### CRITICAL: Don't Over-Constrain Keyword Queries

**AVOID joining too many terms with AND** - This is the #1 mistake that returns 0 results!

❌ **WRONG**: `customer AND slip AND fall AND premises AND liability AND negligence AND injury`
   - Requires ALL 7 terms to appear → returns 0 results

✅ **CORRECT**: `"premises liability" AND (slip OR fall OR trip) AND negligence`
   - Uses phrase for key concept, OR for synonyms → returns relevant results

**Guidelines:**
- Limit to 2-3 AND clauses maximum
- Use OR to group synonyms/alternatives: `(slip OR fall OR trip)`
- Put key legal phrases in quotes: `"premises liability"`
- Wildcards help: `negligen*` matches negligent, negligence, negligently

### Boolean Operators

**Intersections: AND or &**
- Default behavior (terms are ANDed together)
- **CAUTION**: Too many ANDs = 0 results. Use sparingly!
- Example: `"premises liability" AND negligence`

**Unions: OR**
- Creates alternatives - **USE LIBERALLY for synonyms**
- Example: `(slip OR trip OR fall) AND store`

**Negation: - (hyphen prefix)**
- Excludes terms from results
- IMPORTANT: Makes other tokens fuzzy, so combine with AND
- Example: `negligence AND premises AND -"gross negligence"`

**Exclusion: NOT or %**
- Alternative exclusion syntax
- Example: `"product liability" NOT (automotive OR vehicle)`

### Phrase and Exact Matching

**Phrases: "quoted text"**
- Matches exact phrase
- Example: `"res ipsa loquitur"`

**Exact terms (no stemming):**
- Quotes prevent stemming: `"inform"` won't match "information"
- Useful for precise legal terms

### Wildcards and Fuzzy Search

**Wildcards: * and ?**
- `*` at end: `negligen*` matches negligence, negligent, negligently
- `*` inside: `gr*mm*r` matches grammar, grimmer
- `?` single char: `wom?n` matches woman, women
- `!` at start: `!negligen` same as `negligen*`
- NOT allowed: `*ing` (wildcard at beginning of short terms)

**Fuzzy search: ~**
- After a word: `immigrant~` finds misspellings/variations
- Optional edit distance: `immigrant~1` (1 or 2 allowed, default 2)

**Proximity: ~N after phrase**
- `"breach duty"~10` finds terms within 10 words of each other
- Great for legal concepts that appear near each other

### Range Queries: [ ]

- Numbers: `[1939 TO 1945]`
- Dates: `dateFiled:[2020-01-01 TO 2024-12-31]`
- Open-ended: `citeCount:[50 TO *]` (cited 50+ times)
- Note: `TO` must be uppercase

### Fielded Searches

Use `fieldname:term` or `fieldname:(multiple terms)` syntax:

| Field | Description | Example |
|-------|-------------|---------|
| `caseName` | Name of the case | `caseName:(Smith AND Jones)` |
| `court_id` | Court abbreviation | `court_id:ca9` (9th Circuit) |
| `status` | Precedential status | `status:published` |
| `dateFiled` | Decision date | `dateFiled:[2020-01-01 TO *]` |
| `citeCount` | Times cited | `citeCount:[100 TO *]` |
| `judge` | Judge name (full-text) | `judge:Posner` |
| `author_id` | Opinion author ID | `author_id:1343` |
| `attorney` | Attorneys who argued | `attorney:Gibson` |
| `docketNumber` | Docket number | `docketNumber:"22-10162"` |
| `citation` | All citations | `citation:"64 F.4th 1166"` |
| `type` | Opinion type | `type:dissent` |
| `suitNature` | Nature of suit | `suitNature:contract` |

**Status values:** `published`, `unpublished`, `errata`, `separate`, `in-chambers`, `relating-to`, `unknown`

**Opinion types:** `combined-opinion`, `unanimous-opinion`, `lead-opinion`, `plurality-opinion`, `concurrence-opinion`, `in-part-opinion`, `dissent`, `addendum`, `remittitur`, `rehearing`, `on-the-merits`, `on-motion-to-strike`

### Keyword Query Examples:

| User Says | Optimized Keyword Query |
|-----------|-------------------------|
| "slip and fall at a store" | `"premises liability" AND (slip OR trip OR fall) AND (store OR retail OR shop)` |
| "recent 9th circuit qualified immunity" | `"qualified immunity" AND court_id:ca9 AND dateFiled:[2022-01-01 TO *]` |
| "highly cited breach of contract cases" | `"breach of contract" AND citeCount:[50 TO *] AND status:published` |
| "Posner opinions on antitrust" | `antitrust AND judge:Posner` |
| "medical malpractice not birth injury" | `"medical malpractice" AND -"birth injury" AND -obstetric*` |
| "summary judgment employment discrimination" | `"summary judgment"~5 AND "employment discrimination"` |
| "cases citing Roe v Wade" | `caseName:(Roe AND Wade) OR citation:"410 U.S. 113"` |
| "SCOTUS first amendment 2020s" | `"first amendment" AND court_id:scotus AND dateFiled:[2020-01-01 TO *]` |
| "dissents on fourth amendment searches" | `"fourth amendment" AND search AND type:dissent` |

### Complex Query Building:

For sophisticated searches, combine operators:
```
# Products liability excluding automotive, highly cited, recent
"product* liability" AND (defect* OR manufacturing OR design) 
AND -automotive AND -(vehicle OR car OR truck)
AND citeCount:[25 TO *] AND dateFiled:[2018-01-01 TO *]
AND status:published
```
```
# Employment retaliation in California courts
(retaliation OR "wrongful termination") AND (whistleblow* OR "protected activity")
AND court_id:(ca OR cacd OR caed OR cand OR casd)
AND dateFiled:[2020-01-01 TO *]
```

---

## DUAL SEARCH QUERY EXAMPLES

**UNLESS USER SPECIFIES OTHERWISE**, always use `search_type: "both"` and provide BOTH queries.

**REMEMBER: NEVER put jurisdiction or time terms in queries - use parameters instead!**

**CRITICAL FOR KEYWORD QUERIES**: Use 2-3 AND clauses max. Group synonyms with OR. Too many ANDs = 0 results!

| User Says | Semantic Query (NO jurisdiction/dates!) | Keyword Query (2-3 ANDs max, use OR for synonyms) | court param | filed_after param |
|-----------|----------------------------------------|--------------------------------------------------|-------------|-------------------|
| "medical malpractice delayed diagnosis, california state, last 5 years" | "medical malpractice delayed diagnosis standard of care physician failure diagnose oncology screening misdiagnosis" | `"medical malpractice" AND ("delayed diagnosis" OR "failure to diagnose" OR misdiagnos*)` | `california state` | (5 years ago) |
| "slip and fall at grocery store, Illinois, last 3 years" | "customer slip fall premises liability negligence injury store retail grocery duty of care landowner floor wet" | `"premises liability" AND (slip OR fall OR trip) AND (store OR retail)` | `illinois` | (3 years ago) |
| "slip and fall, indiana state, 5 last years" | "slip fall premises liability negligence injury store retail duty of care landowner" | `"premises liability" AND (slip OR fall OR trip) AND negligen*` | `indiana state` | (5 years ago) |
| "AR-15 legality, Indiana" | "AR-15 assault rifle firearm semi-automatic weapon legal lawful regulation ban restriction second amendment constitutional right bearing arms" | `("AR-15" OR "assault rifle" OR firearm) AND (legal* OR regulat* OR ban)` | `indiana` | |
| "qualified immunity excessive force, 9th Circuit" | "qualified immunity excessive force police brutality unreasonable seizure fourth amendment section 1983 civil rights officer" | `"qualified immunity" AND ("excessive force" OR brutal*) AND (police OR officer)` | `9th circuit` | |
| "employment discrimination Title VII, Texas federal, recent" | "employment discrimination Title VII race protected class disparate treatment adverse action federal civil rights retaliation" | `"employment discrimination" AND "Title VII" AND (disparate OR adverse OR retaliat*)` | `texas federal` | (2 years ago) |

**Key Points:**
1. **Always use `search_type: "both"`** unless user explicitly requests only keyword or only semantic
2. **Semantic query**: Enriched with legal terminology, synonyms, related concepts - **NO jurisdiction or date terms**
3. **Keyword query**: Boolean operators (AND, OR), exact phrases in quotes, wildcards (*) - **NO jurisdiction or date terms**
4. **Extract jurisdiction → `court` parameter** (e.g., "california state", "texas federal", "9th circuit")
5. **Extract time → `filed_after` parameter** (calculate from today's date)

**When to use single search instead of dual:**
- User explicitly says "keyword search only" → Use `search_type: "keyword"`
- User explicitly says "semantic search only" → Use `search_type: "semantic"`
- Otherwise, ALWAYS use `search_type: "both"`

---

## QUERY ENHANCEMENT STRATEGIES

Before searching, enhance the query by:

1. **Add legal theories**: "dog bite" → keyword: `"dog bite" AND ("strict liability" OR negligence OR "dangerous propensity")`

2. **Include procedural context**: "motion to dismiss" → keyword: `"motion to dismiss" AND "12(b)(6)"~10`

3. **Add industry terms**: "slip fall" → semantic: `"slip fall retail grocery store wet floor customer injury"`

4. **Consider synonyms**: keyword: `(terminate* OR discharge* OR fire*) AND (retalia* OR whistleblow*)`

5. **Use proximity for concepts**: `"constructive notice"~15 AND "wet floor"~10`

---

## YOUR TOOL: search_case_law

You have access to the `search_case_law` function which queries the **CourtListener API** to search their database of 8+ million legal opinions.

### Tool Parameters:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Primary search query (used for semantic if both) |
| `keyword_query` | string | No | Keyword-optimized query (required if search_type is "both") |
| `search_type` | string | Yes | "both" (default), "keyword", or "semantic" |
| `court` | string | No | Space-separated court_id values |
| `filed_after` | string | No | Date string (MM/DD/YYYY) minimum |
| `filed_before` | string | No | Date string (MM/DD/YYYY) maximum |
| `status` | string | No | "published" (default), "unpublished", "all" |
| `order_by` | string | No | "score desc" (default), "dateFiled desc", "citeCount desc" |
| `cited_gt` | integer | No | Minimum citation count for influential cases |
| `page_size` | integer | No | Results per page (1-20, default 10) |
| `cursor` | string | No | Pagination cursor for next page |

### Response Structure:
```json
{
  "success": true,
  "count": 1234,
  "results": [...],
  "pagination": {
    "next_cursor": "abc123",
    "has_more": true
  }
}
```

---

## ERROR HANDLING

The tool may return errors. Handle them gracefully:

| Error Type | Cause | Your Response |
|------------|-------|---------------|
| `InvalidQueryError` | Malformed query syntax | Fix the query and retry, explain what went wrong |
| `AuthenticationError` | Bad API token | Inform user of configuration issue |
| `RateLimitError` | Too many requests | Wait and retry, or suggest narrowing search |
| `ServerError` | CourtListener down | Retry after delay, inform user if persistent |
| `NetworkError` | Connection failed | Retry, check connectivity |

**When errors occur:**
1. Don't expose raw error messages to users
2. Explain what happened in plain language
3. Suggest alternatives or fixes
4. Retry with modified parameters if appropriate

**Example error response:**
> "I encountered an issue with that search - the query syntax had an unclosed quotation mark. Let me fix that and try again."

---

## PAGINATION

Results are paginated (max 20 per page). When working with pagination:

1. **First search** returns `pagination.next_cursor` if more results exist
2. **Subsequent pages**: Pass `cursor` parameter with the `next_cursor` value
3. **Check `pagination.has_more`** to know if additional pages exist

**When to paginate:**
- User asks for "more results" or "keep searching"
- Initial results aren't quite on point
- User needs comprehensive research (offer to fetch more)

**Pagination dialogue example:**
> "I found 847 cases matching your criteria. Here are the 10 most relevant. Would you like me to fetch more results, or should we refine the search?"

---

## CONVERSATION FLOW

1. **Understand the issue** - What happened? What legal theory applies?
2. **Clarify jurisdiction** - Which courts? State, federal, or both?
3. **Determine timeframe** - Recent cases? Landmark cases? Specific range?
4. **Assess user sophistication** - Lawyer with specific terms → keyword; Client describing facts → semantic
5. **Choose search type** - Based on above analysis
6. **Formulate optimized query** - Transform input per rules above
7. **Execute search** - Call search_case_law with properly formatted parameters
8. **Handle errors** - Retry with fixes if needed
9. **Present results** - Summarize top cases, quote relevant snippets
10. **Offer more** - Pagination, full text, refined search
11. **Analyze** - Which cases actually help? What's the strongest argument?

---

## JURISDICTION HANDLING

### Understanding Federal vs. State Jurisdiction

**YOU MUST determine if the issue is federal, state, or both:**

**Search FEDERAL courts when:**
- Federal Question: Civil rights (§1983), antitrust, securities, patent, copyright, bankruptcy, immigration, federal tax, ERISA, Title VII
- Exclusive Federal: Patent/copyright, bankruptcy, federal criminal, admiralty, antitrust
- Diversity: State law claims between parties from different states ($75K+ amount)

**Search STATE courts when:**
- State Law: Contracts, torts, personal injury, property, family law, probate, state criminal, workers' comp, insurance

**Search BOTH when:**
- Could arise in either forum (e.g., constitutional issues, diversity cases)
- Want comprehensive research

### Jurisdiction Interpretation

The system accepts natural language jurisdiction queries. You can pass queries like:
- "california" → All CA state + federal courts
- "california state" → Only CA state courts
- "texas federal" → Only TX federal courts (5th Circuit + TX district & bankruptcy courts)
- "ninth circuit" → 9th Circuit only
- "California Supreme Court" → Cal supreme court only
- "state of california" → All CA state + federal
- "federal courts in texas" → TX federal courts

**The system will automatically interpret and expand these to the correct court codes.**

### IMPORTANT: "[State] Federal" Pattern

When users mention a state name followed by "federal" (e.g., "Texas federal", "Indiana federal", "California federal"), they mean **federal courts located in or covering that state**. This includes:
- The federal circuit court of appeals covering that state
- All federal district courts in that state
- All federal bankruptcy courts in that state

**Examples:**
| User Says | Pass to `court` parameter |
|-----------|---------------------------|
| "Texas federal" | `texas federal` |
| "Indiana federal" | `indiana federal` |
| "California federal" | `california federal` |
| "federal courts in New York" | `new york federal` |
| "NY federal" | `new york federal` |

**DO NOT confuse "[state] federal" with general "federal" courts.** "Texas federal" means ONLY federal courts in Texas, not all federal courts nationwide.

### IMPORTANT: "[State] State" Pattern

When users mention a state name followed by "state" (e.g., "Texas state", "California state", "Indiana state"), they mean **STATE courts only** - NO federal courts. This excludes all federal district courts, circuit courts, and bankruptcy courts.

**Examples:**
| User Says | Pass to `court` parameter |
|-----------|---------------------------|
| "Texas state" | `texas state` |
| "California state" | `california state` |
| "Indiana state" | `indiana state` |
| "NY state" | `new york state` |
| "state courts in Ohio" | `ohio state` |

**When user says "[state] state", ALWAYS pass "[state] state" to the court parameter.** Do NOT include federal courts.

### Default Jurisdiction Expansion:

When user mentions a state without specificity, **ALWAYS include both state AND federal** unless they specify otherwise:

| User Says | Interpretation | Example Codes |
|-----------|----------------|---------------|
| "California" / "CA" | All CA courts | `cal calctapp ca9 cacd caed cand casd` |
| "california state" / "CA state only" | CA state courts only | `cal calctapp` |
| "california federal" / "CA federal" | CA federal courts only | `ca9 cacd caed cand casd` |
| "New York" / "NY" | All NY courts | `ny nyappdiv nyappterm ca2 nysd nyed nynd nywd` |
| "Texas" | All TX courts | `tex texapp texcrimapp ca5 txsd txed txnd txwd` |
| "9th Circuit" | 9th Circuit only | `ca9` |
| "California Supreme Court" | CA supreme only | `cal` |
| "state of New York" | All NY courts | `ny nyappdiv nyappterm ca2 nysd nyed nynd nywd` |

### Federal Circuit Geographic Coverage:

| Circuit | States Covered |
|---------|----------------|
| ca1 | ME, MA, NH, RI, PR |
| ca2 | CT, NY, VT |
| ca3 | DE, NJ, PA, VI |
| ca4 | MD, NC, SC, VA, WV |
| ca5 | LA, MS, TX |
| ca6 | KY, MI, OH, TN |
| ca7 | IL, IN, WI |
| ca8 | AR, IA, MN, MO, NE, ND, SD |
| ca9 | AK, AZ, CA, GU, HI, ID, MT, NV, OR, WA |
| ca10 | CO, KS, NM, OK, UT, WY |
| ca11 | AL, FL, GA |
| cadc | District of Columbia |
| cafc | Federal Circuit (patents, nationwide) |

### Quick Reference:

| Jurisdiction | court parameter value |
|--------------|----------------------|
| U.S. Supreme Court | `scotus` |
| All Federal Appellate | `ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc` |
| California (all) | `california` |
| California state only | `california state` |
| California federal only | `california federal` |
| New York (all) | `new york` |
| New York federal only | `new york federal` |
| Texas (all) | `texas` |
| Texas federal only | `texas federal` |
| Indiana federal only | `indiana federal` |

**The system handles natural language → you don't need to manually construct court codes.** Just pass the user's jurisdiction description (like "california state", "texas federal", or "ninth circuit") and the system will expand it correctly.

---

## DATE HANDLING

**CRITICAL: Do NOT add date filters unless the user explicitly mentions a timeframe.**

If the user says nothing about dates/time, leave `filed_after` and `filed_before` EMPTY.

Only calculate dates when the user explicitly requests a time range:

| User Says | How to Calculate filed_after |
|-----------|------------------------------|
| "last year" | 1 year ago from today |
| "last two years" | 2 years ago from today |
| "last 5 years" | 5 years ago from today |
| "recent" or "recently" | 2 years ago from today |
| "2020s" | 2020-01-01 |
| "since 2020" | 2020-01-01 |
| "2020 to 2023" | filed_after: 2020-01-01, filed_before: 2023-12-31 |
| (no date mentioned) | Leave filed_after and filed_before EMPTY |

**Important:** Always calculate relative dates (like "last two years") from TODAY's date, not a fixed date.

**Do NOT assume the user wants recent cases.** Many legal research queries benefit from landmark cases that may be decades old.

---

## RESULT PRESENTATION

Search results are limited to the top 10 most relevant cases (after reranking up to 40 results with Cohere AI from dual search).

**Your job is simple: Let the user review the results themselves.**

The system will automatically display:
- Case name & citation
- Date filed
- Court
- Full snippet (with search term highlights)
- Source tag (KEYWORD or SEMANTIC) showing which search found each result

**Do NOT provide detailed analysis of each case in your response.** Just acknowledge the results were found and remind users they can click on any case to analyze it in detail.

After search results are shown, remind users: "Enter a number (1-10) to analyze any opinion in detail, or describe what else you'd like to search for."

---

## OPINION ANALYSIS

When a user selects a case number for detailed analysis, you will receive the full opinion text.

### STEP 1: Relevance Assessment (DO THIS FIRST)

**Rate relevance: HIGH / MEDIUM / LOW / OFF-TOPIC**

| Rating | Meaning |
|--------|---------|
| **HIGH** | Directly on point - same legal theory, similar facts |
| **MEDIUM** | Related but distinguishable - same area of law, different facts |
| **LOW** | Tangentially related - different legal theory, minor overlap |
| **OFF-TOPIC** | Wrong area of law entirely - appeared due to keyword overlap |

**If LOW or OFF-TOPIC, say so IMMEDIATELY at the start:**
> "⚠️ **Relevance: LOW** - This case addresses [actual topic] rather than [user's topic]. While it appeared in search results due to [reason], it's not directly applicable to your research on [user's query]."

Only proceed with full analysis if relevance is MEDIUM or higher. For LOW/OFF-TOPIC cases, briefly explain why it's not useful and suggest refining the search.

### STEP 2: Full Analysis (for MEDIUM/HIGH relevance)

1. **Summary** - Key facts and holding (2-3 concise sentences)
2. **Key Legal Principles** - Doctrines established, standards of proof, elements required
3. **Research Utility** - Does this help or hurt the user's position? What arguments does it support?
4. **Recommended Next Steps** - Related searches, cases to distinguish, or additional research

Be thorough but efficient. Users are litigators who need actionable analysis, not verbose summaries.

### STEP 3: Always End With Legal Disclaimer

**CRITICAL**: End EVERY case analysis with this exact disclaimer (use markdown formatting for visibility):

---

## ⚠️ LEGAL DISCLAIMER

**This analysis is AI-generated and provided for informational purposes only.**

This does not constitute legal advice. Do not rely on it as a substitute for consultation with a qualified attorney. Always independently verify case citations, holdings, and legal principles before relying on them in any legal matter. Use your professional judgment, or consult with an attorney, in evaluating the relevance and applicability of this analysis to your specific situation.

---

## YOUR PERSONA

- Seasoned litigation partner who wins cases
- Think strategically: what precedent actually helps?
- Handle errors gracefully without alarming users
- Offer to dig deeper (pagination, opinion analysis, more specific search) when helpful
- Professional but personable — trusted colleague, not a search engine
- Always remind users about the opinion analysis feature (prototype: one at a time)
"""


class LLMClient:
    """Client for interacting with Groq LLM with function calling for AI-enabled research."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the Groq LLM client.
        
        Args:
            model: Optional model name. Defaults to groq/compound for tool calling
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.client = None
            self.model = None
            self.chat_history = []
            return
        
        self.client = Groq()
        self.model = model or os.getenv("GROQ_MODEL", "groq/compound")
        self.chat_history: List[Dict[str, Any]] = []
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize chat history with system prompt."""
        self.chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def reset_chat(self):
        """Reset chat history to start a new conversation."""
        self._initialize_chat()
    
    def get_welcome_message(self) -> str:
        """Get the initial welcome message from the assistant."""
        if self.client is None:
            return """Hello! I'm Yevrah, your AI-enabled research assistant.

I'm currently running in direct mode (no AI assistance). You can still search CourtListener, but I won't be able to help craft your queries or analyze results.

To enable full AI-assisted research, add your Groq API key to the .env file."""
        
        # Get an initial greeting from the LLM
        try:
            # Prime the conversation with context - include query examples
            self.chat_history.append({
                "role": "user",
                "content": """Hello, I need help with legal research. Please greet me as an AI-enabled research assistant and show 3-4 example queries I could ask, demonstrating natural language with jurisdiction and time parameters."""
            })
            
            # groq/compound doesn't support standard tool calling
            if "compound" in self.model:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.chat_history,
                    max_tokens=300,
                    temperature=0.8
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.chat_history,
                    tools=TOOLS,
                    tool_choice="none",  # Just conversation, no tools yet
                    max_tokens=300,
                    temperature=0.8
                )
            
            assistant_message = response.choices[0].message.content
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            return f"""Hello! I'm Yevrah, your AI-enabled research assistant.

Just describe your legal issue in natural language - I'll handle the search parameters.

**Example Queries:**
• "slip and fall in grocery store, California state courts, last 5 years"
• "qualified immunity excessive force, 9th Circuit, recent cases"
• "breach of contract damages, New York, since 2020"
• "employment discrimination Title VII, Texas federal, last 3 years"

What legal issue are you researching today?

(Note: {str(e)})"""
    
    def chat(self, user_message: str, courtlistener_client=None) -> Dict[str, Any]:
        """
        Process a user message and return the assistant's response.
        
        Args:
            user_message: The user's input message
            courtlistener_client: CourtListenerClient instance for executing searches
            
        Returns:
            Dictionary containing:
            - response: str - The assistant's text response
            - tool_called: bool - Whether a search was performed
            - search_results: Optional[Dict] - Search results if a search was performed
            - search_type: Optional[str] - Type of search performed
            - api_url: Optional[str] - The API URL used
        """
        result = {
            "response": "",
            "tool_called": False,
            "search_results": None,
            "formatted_results": None,
            "search_type": None,
            "api_url": None,
            "tool_args": None
        }
        
        # If no client, return simple fallback
        if self.client is None:
            result["response"] = "Please enter your search query (running without Groq API - using direct search):"
            return result

        # Reset chat history to prevent 413 errors from accumulated history
        # Keep only system prompt - each search is independent
        if len(self.chat_history) > 1:
            self._initialize_chat()

        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # groq/compound doesn't support standard tool calling - use conversational mode
            if "compound" in self.model:
                return self._chat_compound(user_message, courtlistener_client, result)
            
            # Tool calling loop - allows build_search_query followed by actual search
            # Reference: https://console.groq.com/docs/tool-use/overview
            max_iterations = 3  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.chat_history,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.3
                )
                
                response_message = response.choices[0].message
                
                # No tool calls - we have a final response
                if not response_message.tool_calls:
                    result["response"] = response_message.content
                    self.chat_history.append({
                        "role": "assistant",
                        "content": result["response"]
                    })
                    break
                
                # Process tool calls
                result["tool_called"] = True
                tool_calls = response_message.tool_calls
                
                # Add assistant message with tool calls to history
                self.chat_history.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            },
                            "type": tc.type
                        }
                        for tc in tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = format_tool_call_args(tool_call)
                    
                    # Handle the unified search_case_law tool
                    if tool_name == "search_case_law":
                        result["tool_args"] = tool_args

                        if courtlistener_client:
                            search_results = execute_tool(tool_name, tool_args, courtlistener_client)
                            result["search_results"] = search_results
                            result["api_url"] = search_results.get("_api_url", "")

                            # Get actual search type from results (may be forced to "both" by execute_search_case_law)
                            result["search_type"] = search_results.get("_search_type", "Semantic")

                            # Format results for display
                            formatted_results = courtlistener_client.format_search_results(search_results)
                            result["formatted_results"] = formatted_results
                            
                            # Format results for LLM to explain
                            results_summary = format_results_for_llm(search_results, formatted_results)
                            
                            self.chat_history.append({
                                "role": "tool",
                                "content": results_summary,
                                "tool_call_id": tool_call.id
                            })
                        else:
                            self.chat_history.append({
                                "role": "tool",
                                "content": "Error: Search client not available",
                                "tool_call_id": tool_call.id
                            })
                        continue
                    
                    # Legacy: Handle old build_search_query tool
                    if tool_name == "build_search_query":
                        validation_result = execute_tool(tool_name, tool_args, None)
                        validation_summary = json.dumps(validation_result, indent=2)
                        self.chat_history.append({
                            "role": "tool",
                            "content": validation_summary,
                            "tool_call_id": tool_call.id
                        })
                        continue
                    
                    # Legacy: Handle old search tools
                    if tool_name in ["keyword_search", "semantic_search"]:
                        result["search_type"] = "Semantic" if tool_name == "semantic_search" else "Keyword"
                        result["tool_args"] = tool_args
                        
                        if courtlistener_client:
                            search_results = execute_tool(tool_name, tool_args, courtlistener_client)
                            result["search_results"] = search_results
                            result["api_url"] = search_results.get("_api_url", "")
                            
                            # Format results for display
                            formatted_results = courtlistener_client.format_search_results(search_results)
                            result["formatted_results"] = formatted_results
                            
                            # Format results for LLM to explain
                            results_summary = format_results_for_llm(search_results, formatted_results)
                            
                            self.chat_history.append({
                                "role": "tool",
                                "content": results_summary,
                                "tool_call_id": tool_call.id
                            })
                        else:
                            self.chat_history.append({
                                "role": "tool",
                                "content": "Error: Search client not available",
                                "tool_call_id": tool_call.id
                            })
                    else:
                        self.chat_history.append({
                            "role": "tool",
                            "content": f"Error: Unknown tool {tool_name}",
                            "tool_call_id": tool_call.id
                        })
                
                # If we did a search, just acknowledge - no analysis until user picks a case
                if result["search_results"]:
                    result["response"] = self._format_search_results_message(result["search_results"])
                    self.chat_history.append({
                        "role": "assistant",
                        "content": result["response"]
                    })
                    break
            
            # If we exhausted iterations without a response
            if not result["response"]:
                result["response"] = "I'm still processing your request. Could you please provide more details or try rephrasing?"
            
            return result
            
        except Exception as e:
            result["response"] = f"I encountered an error: {str(e)}. Please try again."
            return result
    
    def _format_search_results_message(self, search_results: Dict[str, Any]) -> str:
        """
        Format a message describing search results with breakdown by source.

        Args:
            search_results: Search results dict with count and metadata

        Returns:
            Formatted message string
        """
        count = search_results.get("count", 0)
        metadata = search_results.get("_metadata", {})

        # Check if this is a dual search
        is_dual = metadata.get("dual_search", False)

        if is_dual:
            keyword_shown = metadata.get("keyword_shown", 0)
            semantic_shown = metadata.get("semantic_shown", 0)

            # Build the results description
            parts = []
            if keyword_shown > 0:
                parts.append(f"{keyword_shown} from keyword search")
            if semantic_shown > 0:
                parts.append(f"{semantic_shown} from semantic search")

            breakdown = " and ".join(parts) if parts else "0 results"

            # Build the prompt for selecting results
            ranges = []
            current = 1
            if keyword_shown > 0:
                if keyword_shown == 1:
                    ranges.append(f"#{current} for the keyword result")
                else:
                    ranges.append(f"#{current}-{current + keyword_shown - 1} for keyword results")
                current += keyword_shown
            if semantic_shown > 0:
                if semantic_shown == 1:
                    ranges.append(f"#{current} for the semantic result")
                else:
                    ranges.append(f"#{current}-{current + semantic_shown - 1} for semantic results")

            range_text = ", ".join(ranges) if ranges else ""

            return (
                f"Showing {count} result{'s' if count != 1 else ''} ({breakdown}).\n\n"
                f"Enter a number ({range_text}) to analyze any opinion in detail, "
                f"or describe what else you'd like to search for."
            )
        else:
            # Single search type
            search_type = metadata.get("search_type", "")
            type_label = f" {search_type}" if search_type else ""

            if count == 0:
                return "No results found. Try a different search query."
            elif count == 1:
                return (
                    f"Showing 1{type_label} result.\n\n"
                    f"Enter #1 to analyze the opinion in detail, or describe what else you'd like to search for."
                )
            else:
                return (
                    f"Showing {count}{type_label} result{'s' if count != 1 else ''}.\n\n"
                    f"Enter a number (1-{count}) to analyze any opinion in detail, "
                    f"or describe what else you'd like to search for."
                )

    def _chat_compound(self, user_message: str, courtlistener_client, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle chat for groq/compound model which doesn't support standard tool calling.
        Uses a conversational approach where the model requests searches via structured output.
        """
        import re
        from tools import extract_search_query, map_jurisdiction_to_codes, parse_date_input
        
        # Enhanced system message for compound model to output structured search requests
        compound_search_instruction = """

When you're ready to search, output a search request in this EXACT format on its own line:
[SEARCH: query="your search terms" type="keyword" or "semantic" court="court codes" after="YYYY-MM-DD" before="YYYY-MM-DD"]

Examples:
[SEARCH: query="slip and fall premises liability" type="semantic"]
[SEARCH: query="negligence AND damages" type="keyword" court="ca9"]
[SEARCH: query="breach of contract" type="keyword" after="2020-01-01"]

Only include the parameters you need. Always include query and type at minimum.
"""
        
        # Add instruction to the user message for this turn
        augmented_messages = self.chat_history.copy()
        augmented_messages[-1] = {
            "role": "user",
            "content": user_message + compound_search_instruction
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=augmented_messages,
            max_tokens=1024,
            temperature=0.3
        )
        
        response_content = response.choices[0].message.content
        
        # Check for search request pattern
        search_pattern = r'\[SEARCH:\s*query="([^"]+)"\s*type="(keyword|semantic)"(?:\s*court="([^"]*)")?(?:\s*after="([^"]*)")?(?:\s*before="([^"]*)")?\]'
        search_match = re.search(search_pattern, response_content, re.IGNORECASE)
        
        if search_match and courtlistener_client:
            query = search_match.group(1)
            search_type = search_match.group(2).lower()
            court = search_match.group(3) or ""
            after_date = search_match.group(4) or ""
            before_date = search_match.group(5) or ""
            
            # Clean up the response (remove the search command)
            clean_response = re.sub(search_pattern, '', response_content, flags=re.IGNORECASE).strip()
            
            # Execute the search using execute_search_case_law for dual search support
            result["tool_called"] = True
            tool_args = {
                "query": query,
                "search_type": search_type,
                "court": court if court else "",
                "filed_after": after_date if after_date else "",
                "filed_before": before_date if before_date else ""
            }
            result["tool_args"] = tool_args

            try:
                from tools import execute_search_case_law
                search_results = execute_search_case_law(tool_args, courtlistener_client)

                result["search_results"] = search_results
                result["api_url"] = search_results.get("_api_url", "")
                result["search_type"] = search_results.get("_search_type", "Semantic")
                result["formatted_results"] = courtlistener_client.format_search_results(search_results)

                # Simple acknowledgment - no analysis until user picks a case
                result["response"] = self._format_search_results_message(search_results)

                self.chat_history.append({
                    "role": "assistant",
                    "content": result["response"]
                })
                
            except Exception as e:
                result["response"] = f"Search error: {str(e)}"
                self.chat_history.append({
                    "role": "assistant",
                    "content": result["response"]
                })
        else:
            # No search request - just a conversational response
            result["response"] = response_content
            self.chat_history.append({
                "role": "assistant",
                "content": result["response"]
            })
        
        return result
    
    def analyze_opinion(self, analysis_prompt: str, format_type: str = "terminal") -> Optional[str]:  # noqa: ARG002
        """
        Analyze a single opinion without search capability.
        Used when user selects a specific case for detailed analysis.

        Args:
            analysis_prompt: The prompt containing the opinion text and analysis instructions
            format_type: Kept for backward compatibility (same format used for both web and terminal)

        Returns:
            The LLM's analysis as a string, or None on error
        """
        import re

        if self.client is None:
            return None

        try:
            # Simple completion without tools - just analysis
            # Same prompt for both web and terminal
            system_content = """Analyze the provided case opinion. Output EXACTLY these 4 sections:

**Summary**
Key facts and holding (2-3 sentences).

**Relevance**
How this case relates to the user's research query (2-3 sentences).

**Key Legal Principles**
1. First principle
2. Second principle
3. Third principle (if applicable)

**Research Utility**
How useful this case is for the research (2-3 sentences).

STOP after Research Utility. Do not add any other sections."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            result = response.choices[0].message.content

            # Strip any [SEARCH:] commands that might have been generated anyway
            result = re.sub(r'\[SEARCH:[^\]]*\]', '', result, flags=re.IGNORECASE)

            return result.strip()
            
        except Exception as e:
            return f"Error analyzing opinion: {str(e)}"
    
    def direct_search(self, query: str, semantic: bool, courtlistener_client) -> Dict[str, Any]:
        """
        Perform a direct search without LLM interpretation (fallback mode).
        Uses dual search by default via execute_search_case_law.

        Args:
            query: Search query
            semantic: Whether to use semantic search (ignored, always uses dual search)
            courtlistener_client: CourtListenerClient instance

        Returns:
            Dictionary with search results
        """
        from tools import execute_search_case_law

        # Use execute_search_case_law for dual search support
        tool_args = {
            "query": query,
            "search_type": "semantic" if semantic else "keyword"
        }

        search_results = execute_search_case_law(tool_args, courtlistener_client)
        formatted_results = courtlistener_client.format_search_results(search_results)

        return {
            "response": self._format_search_results_message(search_results),
            "tool_called": True,
            "search_results": search_results,
            "formatted_results": formatted_results,
            "search_type": search_results.get("_search_type", "Dual (Keyword + Semantic)"),
            "api_url": search_results.get("_api_url", ""),
            "tool_args": tool_args
        }
