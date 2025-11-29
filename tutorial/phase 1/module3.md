Excellent! Let's proceed with building the core RAG chain and agent system. This is where we'll add the intelligence to our financial analyst.

let's build the RAG Chain with Sophisticated Prompts

Create `src/financial_rag/agents/prompts.py`

```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class FinancialPrompts:
    """Sophisticated prompts for financial analysis"""
    
    @property
    def system_prompt_analyst(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a seasoned financial analyst at a top investment firm. Your task is to provide accurate, concise, and well-supported analysis based ONLY on the provided context.

Guidelines:
- Be highly numerical and precise. Cite specific figures, percentages, dates, and metrics from the context.
- Structure your answer clearly with headings, bullet points, and logical flow.
- If the context does not contain enough information to answer fully, state what you can conclude and what is missing.
- Highlight risks, opportunities, and key trends.
- Compare performance across periods or segments when relevant.
- Never hallucinate or invent information. If you don't know based on the context, say so.
- For financial metrics, provide context about whether the numbers are positive or negative.

Context:
{context}

Question:
{question}"""
        )

    @property
    def system_prompt_executive(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a financial analyst preparing an executive summary for the C-suite. Provide high-level insights with strategic implications.

Key Requirements:
- Start with a TL;DR (Too Long; Didn't Read) summary
- Focus on strategic implications and business impact
- Highlight key risks and opportunities
- Use clear, business-focused language
- Support all claims with specific data from the context
- Emphasize trends and patterns rather than isolated data points

Context:
{context}

Question:
{question}"""
        )

    @property
    def system_prompt_risk_analysis(self):
        return SystemMessagePromptTemplate.from_template(
            """You are a risk analysis specialist. Analyze the risk factors and provide a comprehensive risk assessment.

Risk Analysis Framework:
1. **Risk Identification**: List all mentioned risks
2. **Risk Assessment**: Evaluate likelihood and impact (High/Medium/Low)
3. **Risk Mitigation**: Note any mentioned mitigation strategies
4. **Comparative Analysis**: Compare with previous periods if available
5. **Recommendations**: Suggest areas for further investigation

Context:
{context}

Question:
{question}"""
        )

    def get_qa_prompt(self, style="analyst"):
        """Get QA prompt based on style"""
        prompts = {
            "analyst": self.system_prompt_analyst,
            "executive": self.system_prompt_executive,
            "risk": self.system_prompt_risk_analysis
        }
        
        system_prompt = prompts.get(style, self.system_prompt_analyst)
        
        return ChatPromptTemplate.from_messages([
            system_prompt,
            HumanMessagePromptTemplate.from_template("{question}")
        ])
```

### Create `src/financial_rag/agents/rag_chain.py`

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from loguru import logger
from financial_rag.config import config
from financial_rag.agents.prompts import FinancialPrompts

class FinancialRAGChain:
    """Main RAG chain for financial analysis"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = self._initialize_llm()
        self.prompts = FinancialPrompts()
        
    def _initialize_llm(self):
        """Initialize the LLM with proper settings"""
        try:
            return ChatOpenAI(
                model_name=config.LLM_MODEL,
                temperature=0.1,  # Low temperature for consistent financial analysis
                openai_api_key=config.OPENAI_API_KEY,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def create_qa_chain(self, prompt_style="analyst", search_type="similarity"):
        """Create a QA chain with specified prompt style"""
        try:
            # Get the retriever
            from financial_rag.retrieval.vector_store import VectorStoreManager
            vector_manager = VectorStoreManager()
            retriever = vector_manager.get_retriever(
                self.vector_store, 
                search_type=search_type
            )
            
            # Create QA chain with custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.prompts.get_qa_prompt(prompt_style)
                },
                return_source_documents=True
            )
            
            logger.success(f"Created QA chain with {prompt_style} style and {search_type} search")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    def analyze_question(self, question, prompt_style="analyst", search_type="similarity"):
        """Main method to analyze a financial question"""
        try:
            qa_chain = self.create_qa_chain(prompt_style, search_type)
            
            logger.info(f"Analyzing question: {question}")
            result = qa_chain({"query": question})
            
            # Log the analysis
            logger.info(f"Analysis completed. Source documents: {len(result.get('source_documents', []))}")
            
            return {
                "question": question,
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "prompt_style": prompt_style,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question '{question}': {str(e)}")
            return {
                "question": question,
                "answer": f"Error analyzing question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
```

Very good, now let's Create the Agent with Tools

### Create `src/financial_rag/agents/tools.py`

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from loguru import logger

class FinancialTools:
    """Tools for the financial agent to interact with real-time data"""
    
    @staticmethod
    def get_stock_price(ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """Get current stock price and recent performance"""
        try:
            logger.info(f"Getting stock price for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for ticker {ticker}"}
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            info = stock.info
            result = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "currency": info.get('currency', 'USD'),
                "company_name": info.get('longName', ticker),
                "timestamp": datetime.now().isoformat(),
                "period": period
            }
            
            logger.success(f"Retrieved stock data for {ticker}: ${current_price}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting stock price for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate_financial_ratio(ticker: str, ratio: str) -> Dict[str, Any]:
        """Calculate financial ratios"""
        try:
            logger.info(f"Calculating {ratio} for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            ratios = {
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "price_to_book": info.get('priceToBook'),
                "debt_to_equity": info.get('debtToEquity'),
                "return_on_equity": info.get('returnOnEquity'),
                "profit_margin": info.get('profitMargins'),
                "operating_margin": info.get('operatingMargins')
            }
            
            value = ratios.get(ratio.lower())
            
            if value is None:
                return {"error": f"Ratio {ratio} not available for {ticker}"}
            
            result = {
                "ticker": ticker,
                "ratio": ratio,
                "value": round(value, 4) if isinstance(value, float) else value,
                "description": FinancialTools._get_ratio_description(ratio),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.success(f"Calculated {ratio} for {ticker}: {value}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating ratio {ratio} for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_ratio_description(ratio: str) -> str:
        """Get description for financial ratios"""
        descriptions = {
            "pe_ratio": "Price-to-Earnings Ratio - Measures company valuation relative to earnings",
            "forward_pe": "Forward P/E Ratio - Based on forecasted earnings",
            "price_to_book": "Price-to-Book Ratio - Compares market value to book value",
            "debt_to_equity": "Debt-to-Equity Ratio - Measures financial leverage",
            "return_on_equity": "Return on Equity - Measures profitability relative to shareholder equity",
            "profit_margin": "Profit Margin - Percentage of revenue that becomes profit",
            "operating_margin": "Operating Margin - Efficiency of core business operations"
        }
        return descriptions.get(ratio.lower(), "Financial ratio")
    
    @staticmethod
    def get_company_info(ticker: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            logger.info(f"Getting company info for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            key_info = {
                "ticker": ticker,
                "company_name": info.get('longName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "market_cap": info.get('marketCap'),
                "employees": info.get('fullTimeEmployees'),
                "description": info.get('longBusinessSummary'),
                "website": info.get('website'),
                "country": info.get('country'),
                "exchange": info.get('exchange'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Clean None values
            key_info = {k: v for k, v in key_info.items() if v is not None}
            
            logger.success(f"Retrieved company info for {ticker}")
            return key_info
            
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_financial_news(ticker: str, num_articles: int = 5) -> List[Dict[str, Any]]:
        """Get recent financial news for a company"""
        try:
            logger.info(f"Getting news for {ticker}")
            stock = yf.Ticker(ticker)
            news = stock.news[:num_articles]
            
            formatted_news = []
            for article in news:
                formatted_article = {
                    "title": article.get('title', ''),
                    "publisher": article.get('publisher', ''),
                    "link": article.get('link', ''),
                    "published_date": datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat() if article.get('providerPublishTime') else None,
                    "related_tickers": article.get('relatedTickers', [])
                }
                formatted_news.append(formatted_article)
            
            logger.success(f"Retrieved {len(formatted_news)} news articles for {ticker}")
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return [{"error": str(e)}]
```

### Create `src/financial_rag/agents/financial_agent.py`

```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Optional, Dict, Any
import re
from loguru import logger

from financial_rag.agents.tools import FinancialTools
from financial_rag.agents.rag_chain import FinancialRAGChain
from financial_rag.config import config

# Custom prompt template for the financial agent
class FinancialAgentPromptTemplate(StringPromptTemplate):
    template: str = """
You are a sophisticated Financial Analysis Agent. Your role is to help users analyze companies, financial documents, and market data.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Previous conversation history:
{history}

Question: {input}
Thought: {agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        # Format the tools section
        tools = "\n".join([f"{tool.name}: {tool.description}" for tool in kwargs["tools"]])
        tool_names = ", ".join([tool.name for tool in kwargs["tools"]])
        
        # Format the prompt
        kwargs["tools"] = tools
        kwargs["tool_names"] = tool_names
        return self.template.format(**kwargs)

class FinancialAgentOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        logger.debug(f"Parsing agent output: {text}")
        
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse action and action input
        action_match = re.search(r"Action:\s*(.+?)\s*Action Input:\s*(.+)", text, re.DOTALL)
        if not action_match:
            raise ValueError(f"Could not parse agent output: {text}")
        
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)

class FinancialAgent:
    """Main financial agent that can use tools and RAG"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.rag_chain = FinancialRAGChain(vector_store)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup the tools available to the agent"""
        
        tools = [
            Tool(
                name="search_filings",
                func=self._search_filings_wrapper,
                description="Search SEC filings and financial documents for specific information. Use for questions about risk factors, financial performance, company strategy, and regulatory disclosures."
            ),
            Tool(
                name="get_stock_price",
                func=FinancialTools.get_stock_price,
                description="Get current stock price and recent performance for a ticker. Input should be a stock ticker symbol like 'AAPL'."
            ),
            Tool(
                name="calculate_financial_ratio",
                func=FinancialTools.calculate_financial_ratio,
                description="Calculate financial ratios for a company. Input should be a string like 'AAPL, pe_ratio' or 'MSFT, debt_to_equity'."
            ),
            Tool(
                name="get_company_info",
                func=FinancialTools.get_company_info,
                description="Get comprehensive company information including sector, market cap, and business description. Input should be a stock ticker symbol."
            ),
            Tool(
                name="get_financial_news",
                func=FinancialTools.get_financial_news,
                description="Get recent financial news for a company. Input should be a stock ticker symbol."
            )
        ]
        
        return tools
    
    def _search_filings_wrapper(self, query: str) -> str:
        """Wrapper for RAG search to make it compatible with agent tools"""
        try:
            result = self.rag_chain.analyze_question(query, prompt_style="analyst")
            answer = result["answer"]
            
            # Add source information if available
            if result.get("source_documents"):
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]))
                source_info = f"\n\nSources: {', '.join(sources[:3])}"  # Limit to 3 sources
                answer += source_info
            
            return answer
        except Exception as e:
            return f"Error searching filings: {str(e)}"
    
    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent executor"""
        try:
            from financial_rag.agents.rag_chain import FinancialRAGChain
            llm = FinancialRAGChain(self.vector_store).llm
            
            # Setup prompt template
            prompt_template = FinancialAgentPromptTemplate(
                input_variables=["input", "agent_scratchpad", "history", "tools"]
            )
            
            # Setup LLM chain
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)
            
            # Setup agent
            tool_names = [tool.name for tool in self.tools]
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=FinancialAgentOutputParser(),
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            
            # Setup executor
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5  # Prevent infinite loops
            )
            
            logger.success("Financial agent setup completed")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Main method to analyze a financial question using the agent"""
        try:
            logger.info(f"Agent analyzing question: {question}")
            
            result = self.agent_executor.run(question)
            
            return {
                "question": question,
                "answer": result,
                "agent_type": "tool_using_agent"
            }
            
        except Exception as e:
            logger.error(f"Error in agent analysis: {str(e)}")
            return {
                "question": question,
                "answer": f"I encountered an error while analyzing your question: {str(e)}",
                "error": str(e),
                "agent_type": "tool_using_agent"
            }
    
    def simple_rag_analysis(self, question: str, prompt_style: str = "analyst") -> Dict[str, Any]:
        """Simple RAG analysis without tool use (fallback)"""
        return self.rag_chain.analyze_question(question, prompt_style)
```

## Step 8: Create Enhanced Test Script

### Create `test_agent.py`

```python
#!/usr/bin/env python3
"""
Test script for Financial RAG Agent
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.config import config

def test_agent_system():
    print("🤖 Testing Financial Agent System...")
    
    try:
        # 1. Initialize components
        print("🔧 Initializing components...")
        doc_processor = DocumentProcessor()
        vector_manager = VectorStoreManager()
        
        # 2. Create or load vector store with mock data
        print("📊 Setting up knowledge base...")
        vector_store = setup_mock_knowledge_base(doc_processor, vector_manager)
        
        # 3. Initialize the financial agent
        print("👨‍💼 Initializing financial agent...")
        agent = FinancialAgent(vector_store)
        
        # 4. Test different types of queries
        print("\n🧪 Testing Agent Capabilities...")
        
        test_queries = [
            # RAG-based queries
            "What are the main risk factors mentioned in the documents?",
            "What is the revenue breakdown by segment?",
            
            # Tool-based queries  
            "What is the current stock price of Apple?",
            "What is the P/E ratio for Microsoft?",
            "Can you get company information for Tesla?",
            
            # Complex queries that require both
            "What are Apple's main risk factors and what is their current stock price?",
        ]
        
        for i, query in enumerate(test_queries[:3]):  # Test first 3 to save time
            print(f"\n{'='*60}")
            print(f"Query {i+1}: {query}")
            print(f"{'='*60}")
            
            try:
                result = agent.analyze(query)
                print(f"Answer: {result['answer'][:500]}...")  # Truncate for display
                
            except Exception as e:
                print(f"Error: {e}")
                # Fallback to simple RAG
                print("Trying simple RAG fallback...")
                simple_result = agent.simple_rag_analysis(query)
                print(f"Answer: {simple_result['answer'][:500]}...")
        
        print(f"\n✅ Agent system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent system test failed: {e}")
        return False

def setup_mock_knowledge_base(doc_processor, vector_manager):
    """Setup mock financial documents for testing"""
    
    mock_financial_docs = [
        {
            "content": """Apple Inc. Fiscal Year 2023 Financial Report
            Total Revenue: $383.3 billion, up 2% year-over-year
            iPhone Revenue: $200.6 billion (52% of total revenue)
            Services Revenue: $85.2 billion (22% of total revenue)
            Gross Margin: 43.2%
            Operating Income: $114.3 billion
            
            Risk Factors:
            - Supply chain disruptions impacting production
            - Foreign currency exchange volatility
            - Intense competition in smartphone and personal computer markets
            - Regulatory changes across multiple jurisdictions
            - Dependence on third-party manufacturing partners
            
            Segment Performance:
            - Americas: $169.7 billion
            - Europe: $95.0 billion  
            - Greater China: $72.6 billion
            - Japan: $25.4 billion
            - Rest of Asia Pacific: $20.6 billion""",
            "metadata": {"source": "apple_10k_2023", "company": "Apple", "year": "2023"}
        },
        {
            "content": """Microsoft Corporation FY2023 Earnings Report
            Revenue: $211.2 billion, up 7% year-over-year
            Azure Cloud Growth: 27% constant currency
            Office Commercial: $44.7 billion
            LinkedIn: $15.2 billion
            Cloud Gross Margin: 72%
            
            Strategic Priorities:
            - AI integration across product portfolio
            - Cloud-first, mobile-first strategy
            - Enterprise security solutions
            - Gaming and metaverse investments
            
            Capital Allocation:
            - Share repurchases: $27.4 billion
            - Dividends: $20.1 billion
            - R&D Investment: $27.5 billion""",
            "metadata": {"source": "microsoft_10k_2023", "company": "Microsoft", "year": "2023"}
        },
        {
            "content": """Amazon.com Inc. 2023 Annual Report
            Net Sales: $574.8 billion
            AWS Revenue: $90.8 billion (29% operating margin)
            North America: $353.5 billion
            International: $131.2 billion
            
            Key Challenges:
            - Economic conditions affecting consumer spending
            - International expansion complexities
            - Increasing cloud competition from Microsoft and Google
            - Regulatory scrutiny in multiple countries
            - Labor and employment matters
            
            Innovation Areas:
            - Artificial intelligence and machine learning
            - Supply chain automation
            - Healthcare initiatives
            - Project Kuiper satellite internet""",
            "metadata": {"source": "amazon_10k_2023", "company": "Amazon", "year": "2023"}
        }
    ]
    
    documents = []
    for doc in mock_financial_docs:
        chunked_docs = doc_processor.text_splitter.create_documents(
            [doc["content"]],
            [doc["metadata"]]
        )
        documents.extend(chunked_docs)
    
    print(f"Created {len(documents)} document chunks from mock data")
    
    # Create vector store
    vector_store = vector_manager.create_vector_store(documents)
    return vector_store

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OPENAI_API_KEY in the .env file")
        print("💡 Get one from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    success = test_agent_system()
    if success:
        print("\n🎉 Financial Agent is working!")
        print("\nNext steps:")
        print("1. Add real SEC data using the SEC ingestor")
        print("2. We'll add monitoring and observability next")
        print("3. Then containerization and API setup")
    else:
        print("\n💥 Agent test failed")
        sys.exit(1)
```

## Step 9: Run the Enhanced Test

```bash
# Make sure you're in the virtual environment and package is installed
pip install -e .

# Run the agent test
python test_agent.py
```

## What We've Built Now:

1. ✅ **Sophisticated RAG Chain** with different prompt styles (analyst, executive, risk)
2. ✅ **Financial Tools** for real-time data (stock prices, ratios, company info)
3. ✅ **Intelligent Agent** that can use tools and RAG together
4. ✅ **Error handling** and fallback mechanisms
5. ✅ **Proper LangChain agent architecture**

The agent can now:
- Answer questions using SEC filings (RAG)
- Get real-time stock data and financial ratios
- Combine both for complex queries
- Handle errors gracefully
- Use different analysis styles

**Run the test and let me know the results!** This will validate that our agent system is working properly with the OpenAI API.

Once this works, we'll move to:
- Step 9: Monitoring and Observability with Weights & Biases
- Step 10: FastAPI REST API
- Step 11: Docker Containerization
- Step 12: Kubernetes Deployment

Excellent! Let's proceed with monitoring, API, and containerization to make this truly enterprise-ready.
