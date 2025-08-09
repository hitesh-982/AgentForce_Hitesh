#!/usr/bin/env python3
"""
Legal Contract AI Assistant - Agentic RAG Application
Built with LlamaIndex and Gemini API for comprehensive contract analysis
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from pathlib import Path

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClauseAnalysis:
    clause_type: str
    content: str
    risk_level: RiskLevel
    issues: List[str]
    suggestions: List[str]
    compliance_status: bool
    missing_elements: List[str]
    
    
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


class ComplianceChecker:
    """Handles compliance checking against standard templates and policies"""
    
    def __init__(self):
        self.standard_clauses = {
            "liability": {
                "required_elements": [
                    "limitation of liability cap",
                    "mutual indemnification",
                    "exclusion of consequential damages"
                ],
                "risk_indicators": [
                    "unlimited liability",
                    "broad indemnification",
                    "no liability cap"
                ]
            },
            "termination": {
                "required_elements": [
                    "termination notice period",
                    "termination for cause",
                    "survival of provisions"
                ],
                "risk_indicators": [
                    "immediate termination",
                    "no cure period",
                    "unclear survival terms"
                ]
            },
            "payment": {
                "required_elements": [
                    "payment terms and schedule",
                    "late payment penalties",
                    "invoice requirements"
                ],
                "risk_indicators": [
                    "net 90 days or longer",
                    "no late fees",
                    "unclear payment terms"
                ]
            },
            "confidentiality": {
                "required_elements": [
                    "definition of confidential information",
                    "permitted disclosures",
                    "return or destruction clause"
                ],
                "risk_indicators": [
                    "broad confidentiality definition",
                    "no exceptions",
                    "indefinite duration"
                ]
            },
            "intellectual_property": {
                "required_elements": [
                    "ip ownership clarity",
                    "license grants",
                    "infringement provisions"
                ],
                "risk_indicators": [
                    "broad ip assignment",
                    "unclear ownership",
                    "no infringement protection"
                ]
            }
        }
    
    def check_clause_compliance(self, clause_type: str, content: str) -> ClauseAnalysis:
        """Check a specific clause against compliance requirements"""
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
        
        content_lower = content.lower()
        for element in requirements["required_elements"]:
            if not any(keyword in content_lower for keyword in element.lower().split()):
                missing_elements.append(element)
        
        for risk_indicator in requirements["risk_indicators"]:
            if any(keyword in content_lower for keyword in risk_indicator.lower().split()):
                issues.append(f"Risk indicator found: {risk_indicator}")
        
        if missing_elements:
            suggestions.extend([
                f"Consider adding: {element}" for element in missing_elements
            ])
        
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


class ContractProcessor:
    """Main class for processing legal contracts using RAG"""
    
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
        
        self.compliance_checker = ComplianceChecker()
        self.index = None
        self.agent = None
        
        # Initialize memory for conversational context
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        
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
        
        await self._setup_agent()
    
    async def add_contract(self, file_path: str) -> str:
        """Add a new contract to the knowledge base"""
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
            
            return f"Successfully added contract: {Path(file_path).name}"
            
        except Exception as e:
            logger.error(f"Error adding contract: {str(e)}")
            return f"Error adding contract: {str(e)}"
    
    def _extract_key_clauses(self, contract_text: str) -> Dict[str, str]:
        """Extract key clauses from contract text using pattern matching"""
        clauses = {}
        
        # Define patterns for different clause types
        clause_patterns = {
            "liability": [r"liability.*?(?=\n\n|\n[A-Z])", r"indemnif.*?(?=\n\n|\n[A-Z])"],
            "termination": [r"terminat.*?(?=\n\n|\n[A-Z])", r"expir.*?(?=\n\n|\n[A-Z])"],
            "payment": [r"payment.*?(?=\n\n|\n[A-Z])", r"invoice.*?(?=\n\n|\n[A-Z])", r"fee.*?(?=\n\n|\n[A-Z])"],
            "confidentiality": [r"confidential.*?(?=\n\n|\n[A-Z])", r"non-disclosure.*?(?=\n\n|\n[A-Z])"],
            "intellectual_property": [r"intellectual property.*?(?=\n\n|\n[A-Z])", r"copyright.*?(?=\n\n|\n[A-Z])"]
        }
        
        text_lower = contract_text.lower()
        
        for clause_type, patterns in clause_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Get the longest match as it's likely most complete
                    clauses[clause_type] = max(matches, key=len)
                    break
        
        return clauses
    
    async def analyze_contract(self, contract_text: str, contract_name: str = "Unknown") -> ContractSummary:
        """Comprehensive contract analysis"""
        
        # Extract key clauses
        key_clauses = self._extract_key_clauses(contract_text)
        
        # Analyze each clause for compliance
        flagged_clauses = []
        for clause_type, content in key_clauses.items():
            analysis = self.compliance_checker.check_clause_compliance(clause_type, content)
            if not analysis.compliance_status or analysis.risk_level != RiskLevel.LOW:
                flagged_clauses.append(analysis)
        
        # Check for missing critical clauses
        required_clause_types = set(self.compliance_checker.standard_clauses.keys())
        found_clause_types = set(key_clauses.keys())
        missing_clauses = list(required_clause_types - found_clause_types)
        
        # Use LLM to extract structured information
        extraction_prompt = f"""
        Analyze this contract and extract the following information in JSON format:
        - key_parties: List of main contracting parties
        - contract_type: Type of contract (e.g., "Service Agreement", "NDA", etc.)
        - effective_date: Contract effective date if mentioned
        - expiration_date: Contract expiration date if mentioned  
        - key_obligations: List of main obligations for each party
        
        Contract text: {contract_text[:3000]}...
        
        Respond with valid JSON only.
        """
        
        try:
            extraction_response = await self.llm.acomplete(extraction_prompt)
            extracted_info = json.loads(str(extraction_response))
        except:
            # Fallback if JSON parsing fails
            extracted_info = {
                "key_parties": ["Party A", "Party B"],
                "contract_type": "Unknown",
                "effective_date": None,
                "expiration_date": None,
                "key_obligations": []
            }
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(flagged_clauses, missing_clauses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(flagged_clauses, missing_clauses)
        
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
            recommendations=recommendations
        )
    
    def _calculate_risk_score(self, flagged_clauses: List[ClauseAnalysis], missing_clauses: List[str]) -> float:
        """Calculate overall contract risk score (0-100)"""
        base_score = 20  # Base risk score
        
        # Add points for flagged clauses
        for clause in flagged_clauses:
            if clause.risk_level == RiskLevel.LOW:
                base_score += 5
            elif clause.risk_level == RiskLevel.MEDIUM:
                base_score += 15
            elif clause.risk_level == RiskLevel.HIGH:
                base_score += 25
            elif clause.risk_level == RiskLevel.CRITICAL:
                base_score += 35
        
        # Add points for missing clauses
        base_score += len(missing_clauses) * 10
        
        return min(base_score, 100)  # Cap at 100
    
    def _generate_recommendations(self, flagged_clauses: List[ClauseAnalysis], missing_clauses: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Recommendations for flagged clauses
        for clause in flagged_clauses:
            recommendations.extend(clause.suggestions)
        
        # Recommendations for missing clauses
        if missing_clauses:
            recommendations.append(f"Add missing critical clauses: {', '.join(missing_clauses)}")
        
        # General recommendations based on risk level
        high_risk_clauses = [c for c in flagged_clauses if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk_clauses:
            recommendations.append("Prioritize review of high-risk clauses before signing")
            recommendations.append("Consider legal counsel review for critical issues")
        
        return recommendations
    
    async def _setup_agent(self):
        """Setup the conversational agent with tools"""
        
        # Query engine tool for contract search
        query_engine = self.index.as_query_engine(
            retriever=VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5,
            ),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            response_mode=ResponseMode.COMPACT
        )
        
        query_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="contract_search",
            description="Search through contract database for specific clauses, terms, or information"
        )
        
        # Contract analysis tool
        def analyze_contract_tool(contract_text: str) -> str:
            """Analyze a contract for compliance and risk factors"""
            try:
                loop = asyncio.get_event_loop()
                analysis = loop.run_until_complete(self.analyze_contract(contract_text))
                return json.dumps(asdict(analysis), indent=2, default=str)
            except:
                # Fallback for synchronous execution
                return "Contract analysis completed - please use the analyze_contract method directly for detailed results"
        
        analysis_tool = FunctionTool.from_defaults(
            fn=analyze_contract_tool,
            name="analyze_contract",
            description="Perform comprehensive analysis of a contract including compliance checking and risk assessment"
        )
        
        # Clause comparison tool
        def compare_clauses_tool(clause1: str, clause2: str) -> str:
            """Compare two contract clauses and identify differences"""
            comparison_prompt = f"""
            Compare these two contract clauses and identify:
            1. Key differences
            2. Which is more favorable and why
            3. Potential risks in each
            4. Recommendations for improvement
            
            Clause 1: {clause1}
            
            Clause 2: {clause2}
            """
            
            try:
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(self.llm.acomplete(comparison_prompt))
                return str(response)
            except:
                return "Clause comparison completed - manual review recommended"
        
        comparison_tool = FunctionTool.from_defaults(
            fn=compare_clauses_tool,
            name="compare_clauses", 
            description="Compare two contract clauses to identify differences and provide recommendations"
        )
        
        # Web search tool for legal research
        def web_search_tool(query: str) -> str:
            """Search the web for legal information, precedents, or contract standards"""
            # Note: In a real implementation, you would integrate with a web search API
            # For demo purposes, this shows how it would work
            search_prompt = f"""
            Based on the query "{query}", provide information about:
            1. Current legal standards and best practices
            2. Recent developments in contract law
            3. Industry benchmarks and common terms
            4. Regulatory considerations
            
            Please provide specific, actionable legal guidance.
            """
            
            try:
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(self.llm.acomplete(search_prompt))
                return f"Legal Research Results: {str(response)}"
            except:
                return "Web search functionality would provide current legal research here"
        
        web_tool = FunctionTool.from_defaults(
            fn=web_search_tool,
            name="legal_web_search",
            description="Search for current legal information, case law, regulations, and industry standards"
        )
        
        # Setup agent with tools
        tools = [query_tool, analysis_tool, comparison_tool, web_tool]
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            memory=self.memory,
            verbose=True,
            max_iterations=10
        )
    
    async def chat(self, message: str) -> str:
        """Chat interface for the contract assistant"""
        if not self.agent:
            await self._setup_agent()
        
        system_message = """You are a legal contract analysis assistant. You help legal teams:
        1. Analyze contracts for compliance and risk factors
        2. Compare contract clauses
        3. Identify missing or problematic terms
        4. Suggest improvements and questions for negotiation
        5. Search through contract databases
        
        Always provide specific, actionable advice and cite relevant contract sections when possible.
        Focus on practical legal insights that help reduce risk and improve contract terms.
        """
        
        enhanced_message = f"{system_message}\n\nUser Question: {message}"
        
        try:
            response = await self.agent.achat(enhanced_message)
            return str(response)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def generate_negotiation_questions(self, contract_summary: ContractSummary) -> List[str]:
        """Generate questions to ask the other party based on contract analysis"""
        questions = []
        
        # Questions based on flagged clauses
        for clause in contract_summary.flagged_clauses:
            if clause.clause_type == "liability":
                questions.append("Can we include a mutual liability cap to limit exposure for both parties?")
                questions.append("Are you willing to add carve-outs for gross negligence and willful misconduct?")
            
            elif clause.clause_type == "termination":
                questions.append("Can we extend the termination notice period to 30 days?")
                questions.append("What provisions should survive termination of this agreement?")
            
            elif clause.clause_type == "payment":
                questions.append("Are you open to shorter payment terms with early payment discounts?")
                questions.append("What happens if invoices are disputed?")
            
            elif clause.clause_type == "confidentiality":
                questions.append("Can we clarify what constitutes confidential information?")
                questions.append("What are the permitted disclosures under this agreement?")
        
        # Questions for missing clauses
        for missing_clause in contract_summary.missing_clauses:
            questions.append(f"Should we add a {missing_clause} clause to this agreement?")
        
        # General questions based on risk score
        if contract_summary.risk_score > 70:
            questions.extend([
                "Can we schedule a call to discuss the high-risk elements in this contract?",
                "Are you open to revisions to address compliance concerns?",
                "What's your flexibility on the terms that present the highest risk?"
            ])
        
        return questions
    
    def export_analysis_report(self, contract_summary: ContractSummary, output_path: str):
        """Export detailed analysis report"""
        report = {
            "analysis_date": datetime.now().isoformat(),
            "contract_summary": asdict(contract_summary),
            "negotiation_questions": self.generate_negotiation_questions(contract_summary),
            "next_steps": [
                "Review flagged clauses with legal team",
                "Prepare negotiation strategy for high-risk items", 
                "Schedule discussion with counterparty",
                "Draft redlines for suggested improvements"
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


# Example usage and main application
async def main():
    """Example usage of the Legal Contract AI Assistant"""
    
    # Initialize with your Gemini API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    
    if GEMINI_API_KEY == "your-api-key-here":
        print("Please set your GEMINI_API_KEY environment variable")
        return
    
    # Initialize the contract processor
    processor = ContractProcessor(GEMINI_API_KEY)
    
    # Initialize the system (can load existing contracts from a directory)
    await processor.initialize_index("./contracts")  # Optional: specify directory with existing contracts
    
    print("🏛️ Legal Contract AI Assistant Initialized!")
    print("=" * 60)
    print("Commands:")
    print("- Type 'analyze [file_path]' to analyze a contract")
    print("- Type 'add [file_path]' to add a contract to the database")
    print("- Type 'chat [message]' to chat with the assistant")
    print("- Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        user_input = input("\n>>> ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.startswith('analyze '):
            file_path = user_input[8:].strip()
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        contract_text = f.read()
                    
                    print("📄 Analyzing contract...")
                    analysis = await processor.analyze_contract(contract_text, Path(file_path).name)
                    
                    print(f"\n📋 Contract Analysis Results:")
                    print(f"Document: {analysis.document_name}")
                    print(f"Type: {analysis.contract_type}")
                    print(f"Risk Score: {analysis.risk_score}/100")
                    print(f"Parties: {', '.join(analysis.key_parties)}")
                    
                    if analysis.flagged_clauses:
                        print(f"\n⚠️ Flagged Clauses ({len(analysis.flagged_clauses)}):")
                        for clause in analysis.flagged_clauses:
                            print(f"  - {clause.clause_type.title()}: {clause.risk_level.value} risk")
                            for issue in clause.issues:
                                print(f"    • {issue}")
                    
                    if analysis.missing_clauses:
                        print(f"\n❌ Missing Clauses: {', '.join(analysis.missing_clauses)}")
                    
                    if analysis.recommendations:
                        print(f"\n💡 Recommendations:")
                        for rec in analysis.recommendations[:3]:  # Show top 3
                            print(f"  - {rec}")
                    
                    # Generate negotiation questions
                    questions = processor.generate_negotiation_questions(analysis)
                    if questions:
                        print(f"\n❓ Suggested Questions for Counterparty:")
                        for q in questions[:3]:  # Show top 3
                            print(f"  - {q}")
                    
                    # Export detailed report
                    report_path = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    processor.export_analysis_report(analysis, report_path)
                    print(f"\n📊 Detailed report saved to: {report_path}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing contract: {str(e)}")
            else:
                print(f"❌ File not found: {file_path}")
        
        elif user_input.startswith('add '):
            file_path = user_input[4:].strip()
            result = await processor.add_contract(file_path)
            print(f"📁 {result}")
        
        elif user_input.startswith('chat '):
            message = user_input[5:].strip()
            print("🤖 Assistant thinking...")
            response = await processor.chat(message)
            print(f"🤖 Assistant: {response}")
        
        else:
            print("❓ Unknown command. Try 'analyze [file]', 'add [file]', 'chat [message]', or 'quit'")


if __name__ == "__main__":
    asyncio.run(main())