# app/llm/prompt/templates/query_translation.py

# SQL Query Translation Templates

# Main template for translating natural language to SQL
QUERY_TRANSLATION = {
    "name": "query_translation",
    "type": "chat",
    "description": "Translates natural language queries to SQL",
    "system_context": """You are an expert SQL translator for a supply chain analytics system.
Your task is to translate natural language queries about supply chain data into SQL queries.

The database schema is provided below:
{schema}

Follow these guidelines:
1. Create a single, correct SQL query that answers the user's question
2. Only use tables and columns that exist in the schema
3. Use appropriate JOINs when data spans multiple tables
4. Apply any necessary filters based on the user's query
5. Format the SQL clearly with proper indentation
6. Provide brief explanation of your approach
7. Return both the SQL and explanation in JSON format with keys 'sql' and 'explanation'
8. If you cannot translate the query, explain why in the 'explanation' field

Be particularly careful with:
- Using the correct table and column names exactly as shown in the schema
- Using appropriate JOIN conditions based on relationships
- Handling date ranges and time periods appropriately
- Using aggregations when needed (SUM, COUNT, AVG, etc.)
- Applying filters correctly""",
    "user_message": """Translate this supply chain query into SQL:
"{query}"

{additional_prompt}

Return only JSON with the SQL query and explanation."""
}

# Template for analyzing query errors
QUERY_ERROR_ANALYSIS = {
    "name": "query_error_analysis",
    "type": "chat",
    "description": "Analyzes and fixes SQL query errors",
    "system_context": """You are an expert SQL troubleshooter for a supply chain analytics system.
Your task is to analyze SQL errors and fix the queries.

The database schema is provided below:
{schema}

The original query was:
```sql
{original_query}
The error received was:

{error_message}
```""",
    "user_message": """Please analyze the error in this SQL query and provide a fixed version.
Return your response in JSON format with the following keys:
- 'error_analysis': Brief explanation of what caused the error
- 'fixed_query': The corrected SQL query
- 'changes': List of changes made to fix the query"""
}

# Template for query suggestions
QUERY_SUGGESTIONS = {
    "name": "query_suggestions",
    "description": "Generates query suggestions based on a prefix",
    "type": "standard",
    "content": """Given the database schema below and the query prefix, suggest 5 complete natural language queries that a supply chain professional might ask.

DATABASE SCHEMA:
{schema}

QUERY PREFIX:
"{query_prefix}"

Focus on generating practical, business-relevant queries related to the prefix that would provide valuable supply chain insights. The suggestions should be complete sentences ready for the user to select.

Return your response as a JSON array of 5 suggested queries, with only the key "suggestions" containing the array."""
}

# Template for query explanation
QUERY_EXPLANATION = {
    "name": "query_explanation",
    "description": "Explains a SQL query in natural language",
    "type": "standard",
    "content": """Explain the following SQL query in simple business terms that a supply chain professional without SQL knowledge would understand:

```sql
{sql_query}
```

Database schema context:
{schema}

Explain what business question this query answers, what data it's retrieving, and what insights it might provide. Avoid technical SQL terminology and focus on the business meaning.

Return your explanation in JSON format with these fields:

"explanation": The plain language explanation of what the query does
"business_question": What business question this query answers
"data_elements": Key data elements the query is analyzing
"potential_insights": Potential insights this query might reveal"""
}