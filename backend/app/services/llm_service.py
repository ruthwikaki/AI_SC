"""
LLM Service using centralized configuration
"""
from typing import Dict, Any, Optional, List
import logging
from app.llm.config_manager import llm_config
from app.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM interactions"""
    
    def __init__(self):
        self.client = OllamaClient()
        self.config = llm_config
        logger.info(f"LLM Service initialized with model: {self.config.get_model_display_name()}")
    
    async def generate_query(self, 
                           natural_language: str,
                           schema_context: str,
                           examples: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate SQL query from natural language"""
        
        # Build prompt
        prompt = self._build_query_prompt(natural_language, schema_context, examples)
        
        try:
            # Generate response
            response = await self.client.generate(prompt)
            sql_query = self._extract_sql(response.get("response", ""))
            
            return {
                "success": True,
                "query": sql_query,
                "explanation": response.get("response", ""),
                "model": self.config.get_model_name()
            }
            
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": self.config.get_model_name()
            }
    
    async def analyze_data(self, data: List[Dict], question: str) -> Dict[str, Any]:
        """Analyze data and provide insights"""
        
        prompt = f"""
        Analyze this data and answer the question:
        
        Data: {data[:10]}  # First 10 rows
        Total rows: {len(data)}
        
        Question: {question}
        
        Provide insights and recommendations.
        """
        
        try:
            response = await self.client.generate(prompt)
            return {
                "success": True,
                "analysis": response.get("response", ""),
                "model": self.config.get_model_name()
            }
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_query_prompt(self, 
                          natural_language: str, 
                          schema_context: str,
                          examples: Optional[List[Dict]] = None) -> str:
        """Build prompt for query generation"""
        
        prompt_parts = [
            "You are a SQL expert. Generate a PostgreSQL query based on the following:",
            f"\nDatabase Schema:\n{schema_context}",
            f"\nUser Request: {natural_language}"
        ]
        
        if examples:
            prompt_parts.append("\nExamples:")
            for ex in examples[:3]:  # Limit to 3 examples
                prompt_parts.append(f"- Question: {ex.get('question')}")
                prompt_parts.append(f"  SQL: {ex.get('sql')}")
        
        prompt_parts.append("\nGenerate only the SQL query, no explanations.")
        
        return "\n".join(prompt_parts)
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response"""
        
        # Remove markdown code blocks if present
        if "```sql" in response:
            sql = response.split("```sql")[1].split("```")[0].strip()
        elif "```" in response:
            sql = response.split("```")[1].split("```")[0].strip()
        else:
            sql = response.strip()
        
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    async def check_health(self) -> Dict[str, Any]:
        """Check LLM service health"""
        try:
            available = await self.client.check_model_availability()
            return {
                "healthy": available,
                "model": self.config.get_model_name(),
                "base_url": self.config.get_base_url()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

# Global instance
llm_service = LLMService()
# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service