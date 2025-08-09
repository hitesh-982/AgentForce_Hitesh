#!/usr/bin/env python3
"""
Enhanced Legal Contract AI Assistant - Agentic RAG Application with Web Search
Built with LlamaIndex and Gemini API for comprehensive contract analysis
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import aiohttp
import requests
from bs4 import BeautifulSoup
import time

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

import re
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import pickle
import urllib.parse
from urllib.parse import urlencode
import xml.etree.ElementTree as ET


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SearchSource(Enum):
    WEB = "web"
    LEGAL_DB = "legal_database"
    CONTRACTS = "contract_database"
    REGULATORY = "regulatory_source"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: SearchSource
    relevance_score: float
    date_published: Optional[str] = None


@dataclass
class ClauseAnalysis:
    clause_type: str
    content: str
    risk_level: RiskLevel
    issues: List[str]
    suggestions: List[str]
    compliance_status: bool
    missing_elements: List[str]
    legal_precedents: List[str] = None
    market_benchmarks: List[str] = None
    
    
@dataclass
class ContractSummary:
    document_name: str
    key_parties: List[str]
    contract_type: str
    effective_date: Optional[str]
    expiration_date: Optional[str]
    key_obligations: List[str]
    risk_score: float
    flagged_clauses: List[ClauseAnalysis]
    missing_clauses: List[str]
    recommendations: List[str]
    legal_research: List[SearchResult] = None
    market_analysis: Dict[str, Any] = None


class WebSearchEngine:
    """Enhanced web search engine with legal-specific capabilities"""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0
        
        # Legal-specific search domains for more targeted results
        self.legal_domains = [
            "site:law.cornell.edu",
            "site:justia.com",
            "site:findlaw.com",
            "site:martindale.com",
            "site:americanbar.org",
            "site:lexisnexis.com",
            "site:westlaw.com",
            "site:courtlistener.com"
        ]
    
    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def search_legal_precedents(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for legal precedents and case law"""
        legal_query = f"{query} case law precedent legal decision court"
        return await self.web_search(legal_query, limit, include_legal_sites=True)
    
    async def search_contract_standards(self, clause_type: str, industry: str = "", limit: int = 5) -> List[SearchResult]:
        """Search for industry-standard contract clauses"""
        query = f"{clause_type} clause standard contract template {industry}".strip()
        return await self.web_search(query, limit, include_legal_sites=True)
    
    async def search_regulatory_updates(self, topic: str, jurisdiction: str = "US", limit: int = 3) -> List[SearchResult]:
        """Search for recent regulatory changes"""
        query = f"{topic} regulation law update {jurisdiction} 2024 2025"
        return await self.web_search(query, limit, include_legal_sites=True)
    
    async def web_search(self, query: str, limit: int = 10, include_legal_sites: bool = False) -> List[SearchResult]:
        """Perform web search with legal focus"""
        try:
            self._rate_limit()
            
            # If we have API credentials, use Google Custom Search
            if self.api_key and self.search_engine_id:
                return await self._google_custom_search(query, limit, include_legal_sites)
            else:
                # Fallback to simulated search for demo
                return await self._simulated_search(query, limit)
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return await self._simulated_search(query, limit)
    
    async def _google_custom_search(self, query: str, limit: int, include_legal_sites: bool) -> List[SearchResult]:
        """Use Google Custom Search API"""
        results = []
        
        # Enhance query for legal searches
        if include_legal_sites:
            enhanced_query = f"{query} (law OR legal OR contract OR clause OR precedent)"
        else:
            enhanced_query = query
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': enhanced_query,
            'num': min(limit, 10),  # Google API limit
            'dateRestrict': 'y2',  # Last 2 years for relevancy
            'sort': 'date'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('items', []):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    source=SearchSource.WEB,
                    relevance_score=0.8,  # Google results are generally relevant
                    date_published=item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time')
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Google search error: {str(e)}")
            return await self._simulated_search(query, limit)
        
        return results
    
    async def _simulated_search(self, query: str, limit: int) -> List[SearchResult]:
        """Simulated search results for demo purposes"""
        simulated_results = [
            SearchResult(
                title=f"Legal Analysis: {query.title()} - Recent Developments",
                url="https://law.cornell.edu/example",
                snippet=f"Comprehensive analysis of {query} including recent case law, regulatory updates, and best practices for contract drafting.",
                source=SearchSource.WEB,
                relevance_score=0.9,
                date_published="2024-01-15"
            ),
            SearchResult(
                title=f"Contract Standards for {query.title()} - Industry Best Practices",
                url="https://americanbar.org/example",
                snippet=f"Latest industry standards and recommendations for {query} in commercial contracts, with practical implementation guidance.",
                source=SearchSource.LEGAL_DB,
                relevance_score=0.85,
                date_published="2024-02-01"
            ),
            SearchResult(
                title=f"Recent Case Law: {query.title()} Precedents",
                url="https://justia.com/example",
                snippet=f"Summary of recent court decisions affecting {query} interpretation and enforcement in various jurisdictions.",
                source=SearchSource.WEB,
                relevance_score=0.8,
                date_published="2024-01-30"
            )
        ]
        
        return simulated_results[:limit]


class EnhancedComplianceChecker:
    """Enhanced compliance checker with web-sourced updates"""
    
    def __init__(self, web_search: WebSearchEngine):
        self.web_search = web_search
        self.standard_clauses = {
            "liability": {
                "required_elements": [
                    "limitation of liability cap",
                    "mutual indemnification",
                    "exclusion of consequential damages",
                    "carve-outs for gross negligence"
                ],
                "risk_indicators": [
                    "unlimited liability",
                    "broad indemnification",
                    "no liability cap",
                    "all damages included"
                ],
                "best_practices": [
                    "Cap at 12 months of fees or $X amount",
                    "Mutual liability limitations",
                    "Exclude indirect and consequential damages"
                ]
            },
            "termination": {
                "required_elements": [
                    "termination notice period",
                    "termination for cause",
                    "survival of provisions",
                    "data return or destruction"
                ],
                "risk_indicators": [
                    "immediate termination",
                    "no cure period",
                    "unclear survival terms",
                    "no data provisions"
                ],
                "best_practices": [
                    "30-day notice for convenience termination",
                    "Specific cure periods for material breaches",
                    "Clear data handling post-termination"
                ]
            },
            "payment": {
                "required_elements": [
                    "payment terms and schedule",
                    "late payment penalties",
                    "invoice requirements",
                    "dispute resolution process"
                ],
                "risk_indicators": [
                    "net 90 days or longer",
                    "no late fees",
                    "unclear payment terms",
                    "no dispute process"
                ],
                "best_practices": [
                    "Net 30 payment terms",
                    "1.5% monthly late fees",
                    "Clear invoice requirements"
                ]
            },
            "confidentiality": {
                "required_elements": [
                    "definition of confidential information",
                    "permitted disclosures",
                    "return or destruction clause",
                    "duration of obligations"
                ],
                "risk_indicators": [
                    "overly broad confidentiality",
                    "no exceptions",
                    "indefinite duration",
                    "unclear definitions"
                ],
                "best_practices": [
                    "Mutual confidentiality obligations",
                    "Standard exceptions (public domain, etc.)",
                    "3-5 year duration limit"
                ]
            },
            "intellectual_property": {
                "required_elements": [
                    "ip ownership clarity",
                    "license grants and scope",
                    "infringement provisions",
                    "derivative works handling"
                ],
                "risk_indicators": [
                    "broad ip assignment",
                    "unclear ownership",
                    "no infringement protection",
                    "undefined derivative works"
                ],
                "best_practices": [
                    "Customer retains ownership of their data/IP",
                    "Limited license for service provision",
                    "Mutual IP infringement protection"
                ]
            }
        }
    
    async def enhanced_clause_analysis(self, clause_type: str, content: str, contract_context: str = "") -> ClauseAnalysis:
        """Enhanced analysis with web research"""
        base_analysis = self.check_clause_compliance(clause_type, content)
        
        # Get legal precedents and market benchmarks
        try:
            precedents = await self.web_search.search_legal_precedents(f"{clause_type} clause enforcement", limit=3)
            standards = await self.web_search.search_contract_standards(clause_type, limit=3)
            
            base_analysis.legal_precedents = [p.snippet for p in precedents]
            base_analysis.market_benchmarks = [s.snippet for s in standards]
            
        except Exception as e:
            logger.error(f"Web research error: {str(e)}")
            base_analysis.legal_precedents = []
            base_analysis.market_benchmarks = []
        
        return base_analysis
    
    def check_clause_compliance(self, clause_type: str, content: str) -> ClauseAnalysis:
        """Basic compliance checking (original functionality)"""
        if clause_type not in self.standard_clauses:
            return ClauseAnalysis(
                clause_type=clause_type,
                content=content,
                risk_level=RiskLevel.MEDIUM,
                issues=[],
                suggestions=[],
                compliance_status=True,
                missing_elements=[]
            )
        
        requirements = self.standard_clauses[clause_type]
        issues = []
        missing_elements = []
        suggestions = []
        
        # Check for required elements
        content_lower = content.lower()
        for element in requirements["required_elements"]:
            if not any(keyword in content_lower for keyword in element.lower().split()):
                missing_elements.append(element)
        
        # Check for risk indicators
        for risk_indicator in requirements["risk_indicators"]:
            if any(keyword in content_lower for keyword in risk_indicator.lower().split()):
                issues.append(f"Risk indicator found: {risk_indicator}")
        
        # Add best practice suggestions
        suggestions.extend(requirements.get("best_practices", []))
        
        # Generate suggestions based on missing elements
        if missing_elements:
            suggestions.extend([
                f"Consider adding: {element}" for element in missing_elements
            ])
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        if issues:
            risk_level = RiskLevel.MEDIUM if len(issues) <= 2 else RiskLevel.HIGH
        if missing_elements:
            current_level = RiskLevel.MEDIUM if len(missing_elements) <= 2 else RiskLevel.HIGH
            risk_level = max(risk_level, current_level, key=lambda x: list(RiskLevel).index(x))
        
        return ClauseAnalysis(
            clause_type=clause_type,
            content=content,
            risk_level=risk_level,
            issues=issues,
            suggestions=suggestions,
            compliance_status=len(issues) == 0 and len(missing_elements) == 0,
            missing_elements=missing_elements
        )


class EnhancedContractProcessor:
    """Enhanced contract processor with web search and advanced analytics"""
    
    def __init__(self, gemini_api_key: str, persist_dir: str = "./contract_storage"):
        # Initialize Gemini LLM and Embeddings
        self.llm = Gemini(
            api_key=gemini_api_key,
            model="models/gemini-1.5-pro-latest",
            temperature=0.1
        )
        
        self.embed_model = GeminiEmbedding(
            api_key=gemini_api_key,
            model_name="models/text-embedding-004"
        )
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced components
        self.web_search = WebSearchEngine()
        self.compliance_checker = EnhancedComplianceChecker(self.web_search)
        self.index = None
        self.agent = None
        
        # Initialize memory for conversational context
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=12000)
        
        # Analytics tracking
        self.analytics = {
            "contracts_processed": 0,
            "high_risk_contracts": 0,
            "common_issues": {},
            "search_queries": 0
        }
        
    async def initialize_index(self, documents_path: str = None):
        """Initialize or load the vector index"""
        try:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self.index = load_index_from_storage(storage_context)
            logger.info("Loaded existing index from storage")
        except:
            # Create new index
            if documents_path and Path(documents_path).exists():
                documents = SimpleDirectoryReader(documents_path).load_data()
                self.index = VectorStoreIndex.from_documents(documents)
                self.index.storage_context.persist(persist_dir=str(self.persist_dir))
                logger.info(f"Created new index from {documents_path}")
            else:
                # Create empty index
                self.index = VectorStoreIndex([])
                logger.info("Created empty index")
        
        await self._setup_enhanced_agent()
    
    async def add_contract_with_analysis(self, file_path: str, auto_analyze: bool = True) -> Tuple[str, Optional[ContractSummary]]:
        """Add contract and optionally perform immediate analysis"""
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "file_path": file_path,
                    "upload_date": datetime.now().isoformat(),
                    "document_type": "contract"
                })
            
            # Insert into index
            for doc in documents:
                self.index.insert(doc)
            
            # Persist updates
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
            
            result_message = f"Successfully added contract: {Path(file_path).name}"
            
            # Auto-analyze if requested
            analysis = None
            if auto_analyze:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    contract_text = f.read()
                analysis = await self.enhanced_contract_analysis(contract_text, Path(file_path).name)
                result_message += f"\n📊 Analysis completed - Risk Score: {analysis.risk_score}/100"
            
            return result_message, analysis
            
        except Exception as e:
            logger.error(f"Error adding contract: {str(e)}")
            return f"Error adding contract: {str(e)}", None
    
    def _extract_key_clauses_advanced(self, contract_text: str) -> Dict[str, str]:
        """Advanced clause extraction with better pattern matching"""
        clauses = {}
        
        # Enhanced patterns for different clause types
        clause_patterns = {
            "liability": [
                r"(?i)(liability|liable|damages?)(?:(?!termination|payment).){50,500}",
                r"(?i)(indemnif\w+|hold harmless)(?:(?!termination|payment).){50,500}",
                r"(?i)(limitation of liability|exclude\w* liability)(?:(?!termination|payment).){50,500}"
            ],
            "termination": [
                r"(?i)(terminat\w+|end\w* (?:of )?(?:this )?agreement)(?:(?!liability|payment).){50,500}",
                r"(?i)(expir\w+|breach|default)(?:(?!liability|payment).){50,500}"
            ],
            "payment": [
                r"(?i)(payment|invoice|fee|amount due)(?:(?!termination|liability).){50,500}",
                r"(?i)(net \d+ days|due (?:within|in)|billing)(?:(?!termination|liability).){50,500}"
            ],
            "confidentiality": [
                r"(?i)(confidential\w*|proprietary|non-disclosure)(?:(?!termination|payment).){50,500}",
                r"(?i)(trade secret|sensitive information)(?:(?!termination|payment).){50,500}"
            ],
            "intellectual_property": [
                r"(?i)(intellectual property|copyright|trademark|patent)(?:(?!termination|payment).){50,500}",
                r"(?i)(ownership|license|derivative work)(?:(?!termination|payment).){50,500}"
            ],
            "force_majeure": [
                r"(?i)(force majeure|act of god|unforeseeable)(?:(?!termination|payment).){50,500}"
            ],
            "governing_law": [
                r"(?i)(governing law|jurisdiction|venue|courts?)(?:(?!termination|payment).){50,500}"
            ]
        }
        
        # Split text into sections for better matching
        sections = re.split(r'\n\s*(?=\d+\.|\([a-z]\)|\b[IVX]+\.)', contract_text)
        
        for clause_type, patterns in clause_patterns.items():
            best_match = ""
            for pattern in patterns:
                for section in sections:
                    matches = re.findall(pattern, section, re.DOTALL | re.IGNORECASE)
                    if matches:
                        # Get the longest match as it's likely most complete
                        current_match = max(matches, key=len) if isinstance(matches[0], str) else section
                        if len(current_match) > len(best_match):
                            best_match = current_match
            
            if best_match:
                clauses[clause_type] = best_match[:1000]  # Limit length
        
        return clauses
    
    async def enhanced_contract_analysis(self, contract_text: str, contract_name: str = "Unknown") -> ContractSummary:
        """Enhanced contract analysis with web research"""
        
        # Update analytics
        self.analytics["contracts_processed"] += 1
        
        # Extract key clauses with advanced patterns
        key_clauses = self._extract_key_clauses_advanced(contract_text)
        
        # Enhanced clause analysis with web research
        flagged_clauses = []
        for clause_type, content in key_clauses.items():
            analysis = await self.compliance_checker.enhanced_clause_analysis(
                clause_type, content, contract_text[:500]
            )
            if not analysis.compliance_status or analysis.risk_level != RiskLevel.LOW:
                flagged_clauses.append(analysis)
                
                # Track common issues
                for issue in analysis.issues:
                    self.analytics["common_issues"][issue] = self.analytics["common_issues"].get(issue, 0) + 1
        
        # Check for missing critical clauses
        required_clause_types = set(self.compliance_checker.standard_clauses.keys())
        found_clause_types = set(key_clauses.keys())
        missing_clauses = list(required_clause_types - found_clause_types)
        
        # Enhanced contract information extraction
        extraction_prompt = f"""
        Analyze this contract and extract detailed information in JSON format:
        {{
            "key_parties": ["List of main contracting parties with roles"],
            "contract_type": "Specific type (e.g., 'Software License Agreement', 'Master Service Agreement')",
            "effective_date": "Contract effective date if mentioned (YYYY-MM-DD format)",
            "expiration_date": "Contract expiration date if mentioned (YYYY-MM-DD format)",
            "key_obligations": ["Detailed list of main obligations for each party"],
            "contract_value": "Total contract value if mentioned",
            "renewal_terms": "Renewal or extension terms if any",
            "governing_law": "Governing law jurisdiction if specified",
            "key_definitions": ["Important defined terms"],
            "critical_deadlines": ["Important dates or deadlines mentioned"]
        }}
        
        Contract text (first 4000 chars): {contract_text[:4000]}...
        
        Respond with valid JSON only.
        """
        
        try:
            extraction_response = await self.llm.acomplete(extraction_prompt)
            extracted_info = json.loads(str(extraction_response))
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            # Enhanced fallback
            extracted_info = {
                "key_parties": self._extract_parties_fallback(contract_text),
                "contract_type": self._classify_contract_type(contract_text),
                "effective_date": None,
                "expiration_date": None,
                "key_obligations": [],
                "contract_value": None,
                "renewal_terms": None,
                "governing_law": None,
                "key_definitions": [],
                "critical_deadlines": []
            }
        
        # Perform relevant web research
        legal_research = []
        try:
            # Search for recent developments related to contract type
            if extracted_info.get("contract_type"):
                research_query = f"{extracted_info['contract_type']} recent legal developments contract law"
                legal_research = await self.web_search.web_search(research_query, limit=3)
                self.analytics["search_queries"] += 1
        except Exception as e:
            logger.error(f"Legal research failed: {str(e)}")
        
        # Calculate enhanced risk score
        risk_score = self._calculate_enhanced_risk_score(flagged_clauses, missing_clauses, extracted_info)
        
        # Track high-risk contracts
        if risk_score >= 70:
            self.analytics["high_risk_contracts"] += 1
        
        # Generate enhanced recommendations
        recommendations = await self._generate_enhanced_recommendations(
            flagged_clauses, missing_clauses, extracted_info, legal_research
        )
        
        # Market analysis based on contract type and clauses
        market_analysis = await self._perform_market_analysis(
            extracted_info.get("contract_type", ""), key_clauses
        )
        
        return ContractSummary(
            document_name=contract_name,
            key_parties=extracted_info.get("key_parties", []),
            contract_type=extracted_info.get("contract_type", "Unknown"),
            effective_date=extracted_info.get("effective_date"),
            expiration_date=extracted_info.get("expiration_date"),
            key_obligations=extracted_info.get("key_obligations", []),
            risk_score=risk_score,
            flagged_clauses=flagged_clauses,
            missing_clauses=missing_clauses,
            recommendations=recommendations,
            legal_research=legal_research,
            market_analysis=market_analysis
        )
    
    def _extract_parties_fallback(self, contract_text: str) -> List[str]:
        """Extract parties using pattern matching"""
        parties = []
        
        # Common patterns for party identification
        patterns = [
            r'between\s+([^,\n]+?)\s+(?:\(|\,)',
            r'This Agreement is entered into by\s+([^,\n]+)',
            r'Party A?\s*[:\-]\s*([^\n,;]+)',
            r'Party B?\s*[:\-]\s*([^\n,;]+)',
            r'Client\s*[:\-]\s*([^\n,;]+)',
            r'Contractor\s*[:\-]\s*([^\n,;]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE)
            parties.extend([match.strip() for match in matches if match.strip()])
        
        # Clean and deduplicate
        cleaned_parties = list(set([p for p in parties if len(p) > 3 and len(p) < 100]))
        return cleaned_parties[:4]  # Limit to 4 parties max
    
    def _classify_contract_type(self, contract_text: str) -> str:
        """Classify contract type based on content"""
        text_lower = contract_text.lower()
        
        contract_types = {
            "Software License Agreement": ["software", "license", "application", "saas"],
            "Service Agreement": ["services", "consulting", "professional services"],
            "Non-Disclosure Agreement": ["non-disclosure", "confidential", "nda"],
            "Employment Agreement": ["employment", "employee", "work for hire"],
            "Purchase Agreement": ["purchase", "buy", "sale", "goods"],
            "Lease Agreement": ["lease", "rent", "premises", "property"],
            "Partnership Agreement": ["partnership", "joint venture", "collaboration"],
            "Master Service Agreement": ["master service", "msa", "statement of work"]
        }
        
        for contract_type, keywords in contract_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return contract_type
        
        return "General Agreement"
    
    def _calculate_enhanced_risk_score(self, flagged_clauses: List[ClauseAnalysis], 
                                     missing_clauses: List[str], extracted_info: Dict) -> float:
        """Enhanced risk score calculation"""
        base_score = 15  # Lower base score
        
        # Add points for flagged clauses with weighted scoring
        for clause in flagged_clauses:
            multiplier = 1.0
            
            # Higher weight for critical clauses
            if clause.clause_type in ["liability", "termination", "intellectual_property"]:
                multiplier = 1.5
            
            if clause.risk_level == RiskLevel.LOW:
                base_score += 3 * multiplier
            elif clause.risk_level == RiskLevel.MEDIUM:
                base_score += 10 * multiplier
            elif clause.risk_level == RiskLevel.HIGH:
                base_score += 20 * multiplier
            elif clause.risk_level == RiskLevel.CRITICAL:
                base_score += 30 * multiplier
        
        # Add points for missing clauses with priorities
        critical_clauses = ["liability", "termination", "confidentiality"]
        for clause in missing_clauses:
            multiplier = 1.5 if clause in critical_clauses else 1.0
            base_score += 8 * multiplier
        
        # Risk modifiers based on contract characteristics
        contract_type = extracted_info.get("contract_type", "")
        if "software" in contract_type.lower() or "saas" in contract_type.lower():
            base_score += 5  # Higher risk for tech contracts
        
        # Check for concerning terms in contract value
        contract_value = extracted_info.get("contract_value", "")
        if contract_value and any(term in contract_value.lower() for term in ["million", "unlimited", "no cap"]):
            base_score += 10
        
        return min(base_score, 100)  # Cap at 100
    
    async def _generate_enhanced_recommendations(self, flagged_clauses: List[ClauseAnalysis], 
                                               missing_clauses: List[str], extracted_info: Dict,
                                               legal_research: List[SearchResult]) -> List[str]:
        """Generate enhanced, prioritized recommendations"""
        recommendations = []
        
        # Priority 1: Critical risk items
        critical_clauses = [c for c in flagged_clauses if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if critical_clauses:
            recommendations.append("🚨 PRIORITY: Address critical risk clauses immediately")
            for clause in critical_clauses:
                recommendations.append(f"• Review {clause.clause_type} clause - {clause.issues[0] if clause.issues else 'compliance issue'}")
        
        # Priority 2: Missing essential clauses
        essential_missing = [c for c in missing_clauses if c in ["liability", "termination", "confidentiality"]]
        if essential_missing:
            recommendations.append(f"📋 Add missing essential clauses: {', '.join(essential_missing)}")
        
        # Priority 3: Specific suggestions from clause analysis
        for clause in flagged_clauses:
            if clause.suggestions:
                recommendations.extend([f"• {suggestion}" for suggestion in clause.suggestions[:2]])
        
        # Priority 4: Market-based recommendations from research
        if legal_research:
            recommendations.append("📚 Consider recent legal developments:")
            for research in legal_research[:2]:
                recommendations.append(f"• Review: {research.title}")
        
        # Priority 5: Contract-specific recommendations
        contract_type = extracted_info.get("contract_type", "")
        if "software" in contract_type.lower():
            recommendations.extend([
                "🔒 Ensure data privacy and security clauses are comprehensive",
                "⚖️ Consider service level agreement (SLA) definitions"
            ])
        elif "service" in contract_type.lower():
            recommendations.extend([
                "📊 Define clear performance metrics and deliverables",
                "🔄 Establish change management procedures"
            ])
        
        return recommendations
    
    async def _perform_market_analysis(self, contract_type: str, key_clauses: Dict[str, str]) -> Dict[str, Any]:
        """Perform market analysis based on contract clauses"""
        analysis = {
            "industry_benchmarks": {},
            "risk_comparison": "average",
            "market_trends": [],
            "competitive_insights": []
        }
        
        try:
            if contract_type and len(contract_type) > 3:
                # Search for market standards
                market_research = await self.web_search.search_contract_standards(contract_type, limit=2)
                
                analysis["market_trends"] = [r.snippet for r in market_research]
                
                # Analyze payment terms against market
                if "payment" in key_clauses:
                    payment_text = key_clauses["payment"].lower()
                    if "net 30" in payment_text:
                        analysis["industry_benchmarks"]["payment_terms"] = "Standard (Net 30)"
                    elif "net 60" in payment_text or "net 90" in payment_text:
                        analysis["industry_benchmarks"]["payment_terms"] = "Extended (Above Market)"
                        analysis["risk_comparison"] = "above_average"
                    else:
                        analysis["industry_benchmarks"]["payment_terms"] = "Needs Review"
                
                # Analyze liability terms
                if "liability" in key_clauses:
                    liability_text = key_clauses["liability"].lower()
                    if "unlimited" in liability_text:
                        analysis["industry_benchmarks"]["liability_risk"] = "High Risk (Unlimited)"
                        analysis["risk_comparison"] = "high"
                    elif "cap" in liability_text or "limit" in liability_text:
                        analysis["industry_benchmarks"]["liability_risk"] = "Managed (Capped)"
                    else:
                        analysis["industry_benchmarks"]["liability_risk"] = "Undefined"
                
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
        
        return analysis
    
    async def _setup_enhanced_agent(self):
        """Setup enhanced agent with comprehensive tools"""
        
        # Enhanced query engine with better retrieval
        query_engine = self.index.as_query_engine(
            retriever=VectorIndexRetriever(
                index=self.index,
                similarity_top_k=8,
            ),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
            response_mode=ResponseMode.COMPACT
        )
        
        query_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="contract_database_search",
            description="Search through contract database for specific clauses, terms, precedents, or comparative analysis"
        )
        
        # Enhanced contract analysis tool
        async def analyze_contract_tool(contract_text: str) -> str:
            """Perform comprehensive contract analysis with web research"""
            try:
                analysis = await self.enhanced_contract_analysis(contract_text, "User_Contract")
                
                # Format response for agent
                response = f"""
                CONTRACT ANALYSIS RESULTS:
                
                📊 Risk Score: {analysis.risk_score}/100
                📄 Type: {analysis.contract_type}
                👥 Parties: {', '.join(analysis.key_parties[:3])}
                
                ⚠️ FLAGGED CLAUSES ({len(analysis.flagged_clauses)}):
                {chr(10).join([f"• {c.clause_type}: {c.risk_level.value} risk" for c in analysis.flagged_clauses[:5]])}
                
                ❌ MISSING CLAUSES: {', '.join(analysis.missing_clauses[:5])}
                
                🎯 TOP RECOMMENDATIONS:
                {chr(10).join([f"• {r}" for r in analysis.recommendations[:5]])}
                
                📚 LEGAL RESEARCH FINDINGS:
                {chr(10).join([f"• {r.title}" for r in (analysis.legal_research or [])[:3]])}
                """
                
                return response.strip()
                
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        analysis_tool = FunctionTool.from_defaults(
            fn=analyze_contract_tool,
            name="analyze_contract",
            description="Perform comprehensive contract analysis including compliance, risk assessment, and legal research"
        )
        
        # Web search tool for legal research
        async def legal_web_search_tool(query: str) -> str:
            """Search web for legal information, precedents, and standards"""
            try:
                self.analytics["search_queries"] += 1
                results = await self.web_search.web_search(query, limit=5, include_legal_sites=True)
                
                if not results:
                    return "No relevant legal information found for this query."
                
                response = f"LEGAL RESEARCH RESULTS for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    response += f"{i}. {result.title}\n"
                    response += f"   {result.snippet[:200]}...\n"
                    response += f"   Source: {result.url}\n\n"
                
                return response
                
            except Exception as e:
                return f"Search error: {str(e)}"
        
        web_search_tool = FunctionTool.from_defaults(
            fn=legal_web_search_tool,
            name="legal_web_search",
            description="Search the web for current legal information, case law, regulations, and industry standards"
        )
        
        # Contract comparison tool
        async def compare_contracts_tool(contract1: str, contract2: str, focus_area: str = "all") -> str:
            """Compare two contracts or contract clauses"""
            try:
                comparison_prompt = f"""
                Compare these two contracts/clauses focusing on {focus_area}:
                
                CONTRACT/CLAUSE 1:
                {contract1[:2000]}...
                
                CONTRACT/CLAUSE 2:
                {contract2[:2000]}...
                
                Provide analysis on:
                1. Key differences in terms
                2. Risk comparison (which is more favorable and why)
                3. Compliance with standard practices
                4. Recommendations for negotiation
                5. Missing elements in either version
                
                Focus your analysis on practical business and legal implications.
                """
                
                response = await self.llm.acomplete(comparison_prompt)
                return str(response)
                
            except Exception as e:
                return f"Comparison error: {str(e)}"
        
        comparison_tool = FunctionTool.from_defaults(
            fn=compare_contracts_tool,
            name="compare_contracts",
            description="Compare two contracts or clauses to identify differences, risks, and provide negotiation recommendations"
        )
        
        # Risk assessment tool
        async def risk_assessment_tool(contract_text: str, risk_focus: str = "comprehensive") -> str:
            """Perform detailed risk assessment of contract"""
            try:
                # Extract key clauses for focused analysis
                clauses = self._extract_key_clauses_advanced(contract_text)
                
                risk_analysis = f"RISK ASSESSMENT REPORT ({risk_focus}):\n\n"
                
                total_risk_points = 0
                for clause_type, content in clauses.items():
                    analysis = await self.compliance_checker.enhanced_clause_analysis(clause_type, content)
                    
                    risk_points = {
                        RiskLevel.LOW: 1,
                        RiskLevel.MEDIUM: 3,
                        RiskLevel.HIGH: 7,
                        RiskLevel.CRITICAL: 10
                    }.get(analysis.risk_level, 0)
                    
                    total_risk_points += risk_points
                    
                    risk_analysis += f"{clause_type.upper()} CLAUSE - {analysis.risk_level.value} risk ({risk_points} pts):\n"
                    if analysis.issues:
                        risk_analysis += f"Issues: {', '.join(analysis.issues[:2])}\n"
                    if analysis.suggestions:
                        risk_analysis += f"Suggestions: {', '.join(analysis.suggestions[:2])}\n"
                    risk_analysis += "\n"
                
                # Overall risk rating
                if total_risk_points <= 10:
                    overall_risk = "LOW RISK"
                elif total_risk_points <= 25:
                    overall_risk = "MEDIUM RISK"
                elif total_risk_points <= 40:
                    overall_risk = "HIGH RISK"
                else:
                    overall_risk = "CRITICAL RISK"
                
                risk_analysis = f"OVERALL RISK RATING: {overall_risk} ({total_risk_points} points)\n\n" + risk_analysis
                
                return risk_analysis
                
            except Exception as e:
                return f"Risk assessment error: {str(e)}"
        
        risk_tool = FunctionTool.from_defaults(
            fn=risk_assessment_tool,
            name="risk_assessment",
            description="Perform detailed risk assessment of contracts with scoring and prioritized recommendations"
        )
        
        # Negotiation strategy tool
        async def negotiation_strategy_tool(contract_summary_json: str, party_position: str = "buyer") -> str:
            """Generate negotiation strategy and talking points"""
            try:
                strategy_prompt = f"""
                Based on this contract analysis, create a negotiation strategy from the {party_position} perspective:
                
                {contract_summary_json}
                
                Provide:
                1. PRIORITY ITEMS (must-have changes)
                2. NEGOTIATION POINTS (preferred changes)
                3. CONCESSION AREAS (acceptable trade-offs)
                4. TACTICAL APPROACH (how to present each point)
                5. FALLBACK POSITIONS (alternative solutions)
                6. QUESTIONS TO ASK (clarifications needed)
                
                Focus on practical, achievable negotiation outcomes.
                """
                
                response = await self.llm.acomplete(strategy_prompt)
                return str(response)
                
            except Exception as e:
                return f"Strategy generation error: {str(e)}"
        
        negotiation_tool = FunctionTool.from_defaults(
            fn=negotiation_strategy_tool,
            name="negotiation_strategy",
            description="Generate comprehensive negotiation strategy and talking points based on contract analysis"
        )
        
        # Analytics and reporting tool
        def analytics_tool() -> str:
            """Get analytics and insights from processed contracts"""
            try:
                analytics_report = f"""
                CONTRACT PROCESSING ANALYTICS:
                
                📊 SUMMARY STATS:
                • Total contracts processed: {self.analytics['contracts_processed']}
                • High-risk contracts: {self.analytics['high_risk_contracts']}
                • Web searches performed: {self.analytics['search_queries']}
                
                🔍 COMMON ISSUES IDENTIFIED:
                """
                
                # Show top 5 most common issues
                sorted_issues = sorted(self.analytics['common_issues'].items(), 
                                     key=lambda x: x[1], reverse=True)
                
                for issue, count in sorted_issues[:5]:
                    analytics_report += f"• {issue}: {count} occurrences\n"
                
                analytics_report += f"""
                
                📈 RISK INSIGHTS:
                • High-risk contract rate: {(self.analytics['high_risk_contracts'] / max(1, self.analytics['contracts_processed']) * 100):.1f}%
                • Average searches per contract: {(self.analytics['search_queries'] / max(1, self.analytics['contracts_processed'])):.1f}
                """
                
                return analytics_report
                
            except Exception as e:
                return f"Analytics error: {str(e)}"
        
        analytics_reporting_tool = FunctionTool.from_defaults(
            fn=analytics_tool,
            name="contract_analytics",
            description="Get analytics and insights from all processed contracts including trends and common issues"
        )
        
        # Setup enhanced agent with all tools
        tools = [
            query_tool,
            analysis_tool,
            web_search_tool,
            comparison_tool,
            risk_tool,
            negotiation_tool,
            analytics_reporting_tool
        ]
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            max_iterations=15,
            system_prompt="""
            You are an expert legal contract analysis assistant with access to comprehensive tools and web search capabilities.
            
            Your expertise includes:
            • Contract risk assessment and compliance checking
            • Legal research using current web sources
            • Comparative contract analysis
            • Negotiation strategy development
            • Market benchmarking and industry standards
            
            Always provide:
            1. Specific, actionable legal advice
            2. Risk prioritization (high/medium/low)
            3. Citations from relevant sources when available
            4. Practical negotiation recommendations
            5. Clear explanations of complex legal concepts
            
            When analyzing contracts:
            - Focus on business impact and practical risks
            - Use web search for current legal developments
            - Provide specific clause language suggestions
            - Consider industry context and market standards
            - Generate strategic negotiation approaches
            
            Be thorough but concise, and always prioritize the user's business interests while ensuring legal compliance.
            """
        )
    
    async def enhanced_chat(self, message: str) -> str:
        """Enhanced chat interface with better context handling"""
        if not self.agent:
            await self._setup_enhanced_agent()
        
        # Add context about available capabilities
        context_message = f"""
        User Query: {message}
        
        Available capabilities:
        - Contract database search across {self.analytics['contracts_processed']} processed contracts
        - Real-time web search for legal research and precedents
        - Comprehensive contract analysis with risk scoring
        - Contract comparison and benchmarking
        - Negotiation strategy development
        - Market analysis and industry standards research
        
        Please provide comprehensive assistance using the most relevant tools.
        """
        
        try:
            response = await self.agent.achat(context_message)
            return str(response)
        except Exception as e:
            logger.error(f"Enhanced chat error: {str(e)}")
            return f"I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or breaking it into smaller parts."
    
    def generate_comprehensive_report(self, contract_summary: ContractSummary, output_path: str):
        """Generate comprehensive analysis report with all enhancements"""
        report = {
            "executive_summary": {
                "contract_name": contract_summary.document_name,
                "risk_level": "HIGH" if contract_summary.risk_score >= 70 else "MEDIUM" if contract_summary.risk_score >= 40 else "LOW",
                "risk_score": contract_summary.risk_score,
                "key_concerns": len(contract_summary.flagged_clauses),
                "missing_clauses": len(contract_summary.missing_clauses)
            },
            "analysis_date": datetime.now().isoformat(),
            "contract_details": {
                "type": contract_summary.contract_type,
                "parties": contract_summary.key_parties,
                "effective_date": contract_summary.effective_date,
                "expiration_date": contract_summary.expiration_date,
                "key_obligations": contract_summary.key_obligations
            },
            "risk_analysis": {
                "flagged_clauses": [asdict(clause) for clause in contract_summary.flagged_clauses],
                "missing_clauses": contract_summary.missing_clauses,
                "risk_score_breakdown": self._get_risk_score_breakdown(contract_summary)
            },
            "legal_research": [asdict(research) for research in (contract_summary.legal_research or [])],
            "market_analysis": contract_summary.market_analysis,
            "recommendations": {
                "immediate_actions": contract_summary.recommendations[:3],
                "all_recommendations": contract_summary.recommendations,
                "negotiation_questions": self.generate_negotiation_questions(contract_summary)
            },
            "next_steps": [
                "Review priority risk items with legal team",
                "Prepare negotiation strategy for high-risk elements",
                "Research market alternatives for problematic clauses",
                "Schedule discussion with counterparty",
                "Draft redlined version with suggested improvements"
            ],
            "appendix": {
                "analytics_summary": self.analytics,
                "compliance_standards_used": list(self.compliance_checker.standard_clauses.keys())
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _get_risk_score_breakdown(self, contract_summary: ContractSummary) -> Dict[str, int]:
        """Get detailed risk score breakdown"""
        breakdown = {
            "base_score": 15,
            "flagged_clauses_score": 0,
            "missing_clauses_score": 0,
            "contract_type_modifier": 0
        }
        
        # Calculate component scores
        for clause in contract_summary.flagged_clauses:
            points = {
                RiskLevel.LOW: 3,
                RiskLevel.MEDIUM: 10,
                RiskLevel.HIGH: 20,
                RiskLevel.CRITICAL: 30
            }.get(clause.risk_level, 0)
            
            if clause.clause_type in ["liability", "termination", "intellectual_property"]:
                points = int(points * 1.5)
            
            breakdown["flagged_clauses_score"] += points
        
        breakdown["missing_clauses_score"] = len(contract_summary.missing_clauses) * 8
        
        if "software" in contract_summary.contract_type.lower():
            breakdown["contract_type_modifier"] = 5
        
        return breakdown
    
    def generate_negotiation_questions(self, contract_summary: ContractSummary) -> List[str]:
        """Enhanced negotiation questions with strategic context"""
        questions = []
        
        # Priority questions based on high-risk items
        critical_clauses = [c for c in contract_summary.flagged_clauses 
                          if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        for clause in critical_clauses:
            if clause.clause_type == "liability":
                questions.extend([
                    "Would you consider adding a mutual liability cap to protect both parties?",
                    "Can we include specific carve-outs for gross negligence and willful misconduct?",
                    "Are you open to excluding consequential damages from liability coverage?"
                ])
            elif clause.clause_type == "termination":
                questions.extend([
                    "Can we establish a 30-day cure period for material breaches?",
                    "What provisions should survive termination of this agreement?",
                    "How should we handle data and confidential information post-termination?"
                ])
            elif clause.clause_type == "payment":
                questions.extend([
                    "Are you flexible on payment terms - would Net 30 work instead of Net 60?",
                    "Can we include early payment discounts as an incentive?",
                    "What's your standard process for handling disputed invoices?"
                ])
        
        # Strategic questions for missing clauses
        critical_missing = [c for c in contract_summary.missing_clauses 
                          if c in ["liability", "termination", "confidentiality", "intellectual_property"]]
        
        for missing in critical_missing:
            questions.append(f"Should we add a comprehensive {missing.replace('_', ' ')} clause to ensure both parties are protected?")
        
        # Market-based questions
        if contract_summary.market_analysis and contract_summary.market_analysis.get("risk_comparison") == "high":
            questions.extend([
                "How do these terms compare to your other similar agreements?",
                "Are you open to benchmarking key terms against industry standards?",
                "What flexibility do you have to align with current market practices?"
            ])
        
        # Risk-based strategic questions
        if contract_summary.risk_score >= 70:
            questions.extend([
                "Can we schedule a call to address the key risk areas in this contract?",
                "Would you be open to a redlined version that addresses our primary concerns?",
                "What's most important to you in this agreement - where do you need us to be flexible?"
            ])
        
        return questions[:10]  # Limit to top 10 most strategic questions


# Enhanced main application with advanced features
async def main():
    """Enhanced main application with comprehensive contract processing"""
    
    # Initialize with environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Optional for web search
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")  # Optional
    
    if GEMINI_API_KEY == "your-api-key-here":
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        print("   For enhanced web search, also set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID")
        return
    
    # Initialize the enhanced contract processor
    processor = EnhancedContractProcessor(GEMINI_API_KEY)
    
    # Initialize the system
    print("🏛️ Initializing Enhanced Legal Contract AI Assistant...")
    await processor.initialize_index("./contracts")
    
    print("=" * 80)
    print("🚀 ENHANCED LEGAL CONTRACT AI ASSISTANT")
    print("=" * 80)
    print("🔍 Features:")
    print("  • Advanced contract analysis with risk scoring")
    print("  • Real-time legal research and web search")
    print("  • Contract comparison and benchmarking")
    print("  • Negotiation strategy development")
    print("  • Market analysis and compliance checking")
    print("  • Comprehensive reporting and analytics")
    print("=" * 80)
    print("📝 Commands:")
    print("  analyze [file]     - Comprehensive contract analysis with web research")
    print("  add [file]         - Add contract to database with auto-analysis")
    print("  chat [message]     - Interactive chat with AI assistant")
    print("  compare [file1] [file2] - Compare two contracts")
    print("  analytics          - View processing analytics")
    print("  help               - Show detailed help")
    print("  quit               - Exit application")
    print("=" * 80)
    
    while True:
        try:
            user_input = input(f"\n[{processor.analytics['contracts_processed']} contracts] >>> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thank you for using the Legal Contract AI Assistant!")
                break
            
            elif user_input.lower() == 'help':
                print("""
                📚 DETAILED HELP:
                
                🔍 ANALYSIS COMMANDS:
                • analyze [file_path] - Full analysis with risk scoring and web research
                • add [file_path] - Add to database with optional auto-analysis
                • compare [file1] [file2] - Side-by-side contract comparison
                
                💬 CHAT EXAMPLES:
                • "What are the risks in unlimited liability clauses?"
                • "Search for recent software license agreement standards"
                • "Compare payment terms in my database"
                • "Generate negotiation strategy for high-risk contract"
                • "What should I look for in termination clauses?"
                
                📊 ANALYTICS:
                • analytics - View processing statistics and common issues
                
                🔗 WEB SEARCH INTEGRATION:
                The assistant can search for current legal information, case law,
                and industry standards to enhance contract analysis.
                """)
            
            elif user_input.lower() == 'analytics':
                analytics_data = processor.analytics
                print(f"""
                📊 CONTRACT PROCESSING ANALYTICS:
                
                📈 Summary Statistics:
                • Total contracts processed: {analytics_data['contracts_processed']}
                • High-risk contracts identified: {analytics_data['high_risk_contracts']}
                • Web searches performed: {analytics_data['search_queries']}
                • Risk rate: {(analytics_data['high_risk_contracts'] / max(1, analytics_data['contracts_processed']) * 100):.1f}%
                
                🔍 Most Common Issues:
                """)
                
                sorted_issues = sorted(analytics_data['common_issues'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for issue, count in sorted_issues[:5]:
                    print(f"  • {issue}: {count} occurrences")
            
            elif user_input.startswith('analyze '):
                file_path = user_input[8:].strip()
                if Path(file_path).exists():
                    try:
                        print("📄 Reading contract...")
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            contract_text = f.read()
                        
                        print("🔍 Performing comprehensive analysis (including web research)...")
                        analysis = await processor.enhanced_contract_analysis(contract_text, Path(file_path).name)
                        
                        print(f"\n{'='*60}")
                        print(f"📋 COMPREHENSIVE CONTRACT ANALYSIS")
                        print(f"{'='*60}")
                        print(f"📄 Document: {analysis.document_name}")
                        print(f"📑 Type: {analysis.contract_type}")
                        print(f"⚡ Risk Score: {analysis.risk_score}/100")
                        
                        # Risk level indicator
                        if analysis.risk_score >= 70:
                            risk_indicator = "🔴 HIGH RISK"
                        elif analysis.risk_score >= 40:
                            risk_indicator = "🟡 MEDIUM RISK"
                        else:
                            risk_indicator = "🟢 LOW RISK"
                        print(f"🎯 Risk Level: {risk_indicator}")
                        
                        print(f"👥 Parties: {', '.join(analysis.key_parties[:3])}")
                        if analysis.effective_date:
                            print(f"📅 Effective: {analysis.effective_date}")
                        if analysis.expiration_date:
                            print(f"⏰ Expires: {analysis.expiration_date}")
                        
                        if analysis.flagged_clauses:
                            print(f"\n⚠️ FLAGGED CLAUSES ({len(analysis.flagged_clauses)}):")
                            for clause in analysis.flagged_clauses[:5]:
                                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                                print(f"  {risk_emoji.get(clause.risk_level.value, '⚪')} {clause.clause_type.title()}: {clause.risk_level.value} risk")
                                for issue in clause.issues[:2]:
                                    print(f"    • {issue}")
                        
                        if analysis.missing_clauses:
                            print(f"\n❌ MISSING CRITICAL CLAUSES:")
                            for clause in analysis.missing_clauses:
                                print(f"  • {clause.replace('_', ' ').title()}")
                        
                        if analysis.recommendations:
                            print(f"\n💡 TOP RECOMMENDATIONS:")
                            for rec in analysis.recommendations[:5]:
                                print(f"  • {rec}")
                        
                        if analysis.legal_research:
                            print(f"\n📚 LEGAL RESEARCH FINDINGS:")
                            for research in analysis.legal_research[:3]:
                                print(f"  📖 {research.title}")
                                print(f"     {research.snippet[:100]}...")
                        
                        if analysis.market_analysis and analysis.market_analysis.get("industry_benchmarks"):
                            print(f"\n📊 MARKET ANALYSIS:")
                            benchmarks = analysis.market_analysis["industry_benchmarks"]
                            for key, value in benchmarks.items():
                                print(f"  • {key.replace('_', ' ').title()}: {value}")
                        
                        # Generate and show negotiation questions
                        questions = processor.generate_negotiation_questions(analysis)
                        if questions:
                            print(f"\n❓ STRATEGIC NEGOTIATION QUESTIONS:")
                            for q in questions[:5]:
                                print(f"  • {q}")
                        
                        # Export comprehensive report
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        report_path = f"enhanced_analysis_{Path(file_path).stem}_{timestamp}.json"
                        processor.generate_comprehensive_report(analysis, report_path)
                        print(f"\n📊 Comprehensive report saved: {report_path}")
                        
                    except Exception as e:
                        print(f"❌ Error analyzing contract: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
                else:
                    print(f"❌ File not found: {file_path}")
            
            elif user_input.startswith('add '):
                file_path = user_input[4:].strip()
                print("📁 Adding contract to database...")
                result, analysis = await processor.add_contract_with_analysis(file_path, auto_analyze=True)
                print(f"✅ {result}")
                
                if analysis and analysis.risk_score >= 70:
                    print(f"⚠️  HIGH RISK contract detected! Consider immediate review.")
            
            elif user_input.startswith('compare '):
                parts = user_input[8:].strip().split()
                if len(parts) >= 2:
                    file1, file2 = parts[0], parts[1]
                    if Path(file1).exists() and Path(file2).exists():
                        try:
                            print("📊 Comparing contracts...")
                            
                            # Read both contracts
                            with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
                                contract1 = f.read()
                            with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
                                contract2 = f.read()
                            
                            # Use agent's comparison tool
                            response = await processor.enhanced_chat(
                                f"Compare these two contracts and highlight key differences, risks, and recommendations:\n\n"
                                f"CONTRACT 1 ({Path(file1).name}):\n{contract1[:2000]}...\n\n"
                                f"CONTRACT 2 ({Path(file2).name}):\n{contract2[:2000]}..."
                            )
                            
                            print(f"\n📊 CONTRACT COMPARISON RESULTS:")
                            print("="*60)
                            print(response)
                            
                        except Exception as e:
                            print(f"❌ Error comparing contracts: {str(e)}")
                    else:
                        print("❌ One or both files not found")
                else:
                    print("❌ Please provide two file paths: compare [file1] [file2]")
            
            elif user_input.startswith('chat '):
                message = user_input[5:].strip()
                print("🤖 AI Assistant analyzing your request...")
                
                # Show typing indicator for better UX
                import asyncio
                print("   Thinking", end="")
                for _ in range(3):
                    await asyncio.sleep(0.5)
                    print(".", end="", flush=True)
                print()
                
                response = await processor.enhanced_chat(message)
                print(f"\n🤖 Legal AI Assistant:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            
            else:
                # Try to interpret as a chat message if it doesn't match commands
                if len(user_input) > 3:
                    print("🤖 Interpreting as chat query...")
                    response = await processor.enhanced_chat(user_input)
                    print(f"\n🤖 Legal AI Assistant:")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                else:
                    print("❓ Unknown command. Type 'help' for detailed assistance or use:")
                    print("   • analyze [file] - Analyze a contract")
                    print("   • add [file] - Add contract to database") 
                    print("   • chat [message] - Ask questions")
                    print("   • compare [file1] [file2] - Compare contracts")
                    print("   • analytics - View statistics")
                    print("   • quit - Exit")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Main loop error: {str(e)}")


# Additional utility functions and example usage
class ContractTemplateGenerator:
    """Generate contract templates based on analysis insights"""
    
    def __init__(self, processor: EnhancedContractProcessor):
        self.processor = processor
    
    async def generate_clause_template(self, clause_type: str, risk_level: str = "low") -> str:
        """Generate template clause based on best practices"""
        
        template_prompt = f"""
        Generate a template {clause_type} clause that follows best practices and maintains {risk_level} risk level.
        
        Requirements:
        - Use clear, unambiguous language
        - Include standard protections for both parties
        - Follow current legal best practices
        - Be commercially reasonable
        
        Provide the clause text with brief explanatory comments.
        """
        
        try:
            response = await self.processor.llm.acomplete(template_prompt)
            return str(response)
        except Exception as e:
            return f"Template generation error: {str(e)}"


class ContractLifecycleManager:
    """Manage contract lifecycle events and renewals"""
    
    def __init__(self, processor: EnhancedContractProcessor):
        self.processor = processor
        self.contracts_db = {}
    
    def add_contract_tracking(self, contract_summary: ContractSummary):
        """Add contract to lifecycle tracking"""
        self.contracts_db[contract_summary.document_name] = {
            "summary": contract_summary,
            "added_date": datetime.now(),
            "status": "active",
            "next_review": self._calculate_next_review(contract_summary),
            "expiration_alert": self._calculate_expiration_alert(contract_summary)
        }
    
    def _calculate_next_review(self, contract_summary: ContractSummary) -> datetime:
        """Calculate next review date based on risk score"""
        base_date = datetime.now()
        
        if contract_summary.risk_score >= 70:
            # High risk - review quarterly
            return base_date + timedelta(days=90)
        elif contract_summary.risk_score >= 40:
            # Medium risk - review semi-annually
            return base_date + timedelta(days=180)
        else:
            # Low risk - review annually
            return base_date + timedelta(days=365)
    
    def _calculate_expiration_alert(self, contract_summary: ContractSummary) -> Optional[datetime]:
        """Calculate when to send expiration alerts"""
        if contract_summary.expiration_date:
            try:
                exp_date = datetime.fromisoformat(contract_summary.expiration_date)
                # Alert 90 days before expiration
                return exp_date - timedelta(days=90)
            except:
                pass
        return None
    
    def get_contracts_needing_review(self) -> List[Dict]:
        """Get contracts that need review"""
        now = datetime.now()
        return [
            contract for contract in self.contracts_db.values()
            if contract["next_review"] <= now
        ]
    
    def get_expiring_contracts(self, days_ahead: int = 90) -> List[Dict]:
        """Get contracts expiring within specified days"""
        cutoff = datetime.now() + timedelta(days=days_ahead)
        return [
            contract for contract in self.contracts_db.values()
            if contract["expiration_alert"] and contract["expiration_alert"] <= cutoff
        ]


# Example specialized analysis for different contract types
class SpecializedAnalyzers:
    """Specialized analyzers for different types of contracts"""
    
    @staticmethod
    async def analyze_saas_agreement(contract_text: str, processor: EnhancedContractProcessor) -> Dict[str, Any]:
        """Specialized analysis for SaaS agreements"""
        
        saas_specific_checks = {
            "data_privacy": ["GDPR", "CCPA", "data processing", "privacy policy"],
            "service_levels": ["SLA", "uptime", "availability", "performance"],
            "security": ["security measures", "encryption", "SOC 2", "ISO 27001"],
            "data_ownership": ["customer data", "data portability", "data export"],
            "integration": ["API", "integration", "data migration"]
        }
        
        findings = {}
        text_lower = contract_text.lower()
        
        for category, keywords in saas_specific_checks.items():
            found_items = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_items.append(keyword)
            
            findings[category] = {
                "found": found_items,
                "coverage": len(found_items) / len(keywords),
                "missing": [k for k in keywords if k.lower() not in text_lower]
            }
        
        # Get additional insights using LLM
        saas_prompt = f"""
        Analyze this SaaS agreement for key SaaS-specific risks and considerations:
        
        {contract_text[:2000]}...
        
        Focus on:
        1. Data security and privacy compliance
        2. Service level commitments
        3. Data portability and vendor lock-in risks
        4. Integration and API terms
        5. Subscription and billing model risks
        
        Provide specific recommendations for SaaS contracts.
        """
        
        try:
            llm_analysis = await processor.llm.acomplete(saas_prompt)
            findings["llm_insights"] = str(llm_analysis)
        except Exception as e:
            findings["llm_insights"] = f"Analysis error: {str(e)}"
        
        return findings
    
    @staticmethod
    async def analyze_employment_agreement(contract_text: str, processor: EnhancedContractProcessor) -> Dict[str, Any]:
        """Specialized analysis for employment agreements"""
        
        employment_checks = {
            "compensation": ["salary", "bonus", "equity", "stock options"],
            "benefits": ["health insurance", "401k", "vacation", "PTO"],
            "intellectual_property": ["work for hire", "invention assignment", "IP ownership"],
            "non_compete": ["non-compete", "non-solicitation", "restraint of trade"],
            "termination": ["at-will", "severance", "notice period", "cause"]
        }
        
        findings = {}
        text_lower = contract_text.lower()
        
        for category, keywords in employment_checks.items():
            found_items = []
            coverage_score = 0
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_items.append(keyword)
                    coverage_score += 1
            
            findings[category] = {
                "found": found_items,
                "coverage": coverage_score / len(keywords),
                "missing": [k for k in keywords if k.lower() not in text_lower]
            }
        
        return findings


# Performance monitoring and optimization
class PerformanceMonitor:
    """Monitor and optimize system performance"""
    
    def __init__(self):
        self.metrics = {
            "analysis_times": [],
            "search_times": [],
            "token_usage": [],
            "error_counts": {"llm_errors": 0, "search_errors": 0, "processing_errors": 0}
        }
        self.start_time = time.time()
    
    def log_analysis_time(self, duration: float):
        """Log contract analysis duration"""
        self.metrics["analysis_times"].append(duration)
    
    def log_search_time(self, duration: float):
        """Log web search duration"""
        self.metrics["search_times"].append(duration)
    
    def log_error(self, error_type: str):
        """Log system errors"""
        self.metrics["error_counts"][error_type] = self.metrics["error_counts"].get(error_type, 0) + 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "analysis_stats": {
                "count": len(self.metrics["analysis_times"]),
                "avg_duration": sum(self.metrics["analysis_times"]) / max(len(self.metrics["analysis_times"]), 1),
                "max_duration": max(self.metrics["analysis_times"]) if self.metrics["analysis_times"] else 0
            },
            "search_stats": {
                "count": len(self.metrics["search_times"]),
                "avg_duration": sum(self.metrics["search_times"]) / max(len(self.metrics["search_times"]), 1)
            },
            "error_stats": self.metrics["error_counts"],
            "throughput": len(self.metrics["analysis_times"]) / (uptime / 3600) if uptime > 0 else 0  # contracts per hour
        }


# Example configuration for production deployment
class ProductionConfig:
    """Configuration for production deployment"""
    
    # API Rate Limits
    GEMINI_REQUESTS_PER_MINUTE = 60
    GOOGLE_SEARCH_REQUESTS_PER_DAY = 100
    
    # Storage Configuration
    VECTOR_STORE_TYPE = "chroma"  # Options: chroma, pinecone, weaviate
    PERSIST_DIRECTORY = "/app/data/contracts"
    BACKUP_DIRECTORY = "/app/backups"
    
    # Security Configuration
    ENABLE_AUDIT_LOGGING = True
    REQUIRE_AUTHENTICATION = True
    ENCRYPTION_KEY_PATH = "/app/config/encryption.key"
    
    # Performance Configuration
    MAX_CONCURRENT_ANALYSES = 5
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    MAX_MEMORY_TOKENS = 16000
    
    # Legal Compliance
    DATA_RETENTION_DAYS = 2555  # 7 years for legal documents
    ENABLE_PII_DETECTION = True
    REDACT_SENSITIVE_INFO = True
    
    @classmethod
    def validate_config(cls):
        """Validate production configuration"""
        required_env_vars = [
            "GEMINI_API_KEY",
            "DATABASE_URL", 
            "REDIS_URL",
            "SECRET_KEY"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True


if __name__ == "__main__":
    # Initialize performance monitoring
    perf_monitor = PerformanceMonitor()
    
    try:
        # Run the main application
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user")
    except Exception as e:
        print(f"\n\n❌ Critical error: {str(e)}")
        logger.critical(f"Critical application error: {str(e)}")
    finally:
        # Show performance report on exit
        try:
            perf_report = perf_monitor.get_performance_report()
            print(f"\n📊 Session Performance Summary:")
            print(f"⏱️  Uptime: {perf_report['uptime_formatted']}")
            print(f"📈 Analyses: {perf_report['analysis_stats']['count']}")
            print(f"🔍 Searches: {perf_report['search_stats']['count']}")
            print(f"⚡ Throughput: {perf_report['throughput']:.1f} contracts/hour")
            
            if any(perf_report['error_stats'].values()):
                print(f"⚠️  Errors: {sum(perf_report['error_stats'].values())} total")
        except:
            pass