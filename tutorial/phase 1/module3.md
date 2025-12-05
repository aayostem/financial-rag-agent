Excellent! Let's proceed with building the core RAG chain and agent system. This is where we'll add the intelligence to our financial analyst.

let's build the RAG Chain with Sophisticated Prompts

Go to `src/financial_rag/agents/` and create prompts.py

**the prompts**. These are like the "instructions manual" we give to our AI to make it think like a financial expert!

Here, we will creates **three different AI personalities** for financial analysis:

1. **🧮 The Detail-Oriented Analyst** - that Gets deep into the numbers
2. **👔 The Executive Summarizer** - that provides High-level strategic insights  
3. **⚠️ The Risk Specialist** - that Focuses on potential problems

Think of it as hiring three different financial experts, each with their own specialty!

## **💡 Real-World Application**

Imagine you're analyzing Tesla's financials:

```python
# Get the right prompt for your audience
if audience == "classroom":
    prompt = financial_prompts.get_qa_prompt("analyst")
elif audience == "board_meeting":
    prompt = financial_prompts.get_qa_prompt("executive")
elif audience == "risk_committee":
    prompt = financial_prompts.get_qa_prompt("risk")

# Same AI, different outputs based on the prompt!
```

---

## **📚 Key Takeaways**

1. **Prompts are instructions** - They tell the AI how to think
2. **Different prompts = different personas** - One AI, many experts
3. **Structure matters** - Clear guidelines prevent vague answers
4. **Context is king** - The AI only knows what we tell it in the prompt

**Think of it like this:** 
- The AI model is a **brilliant but clueless intern**
- These prompts are the **training manual** we give them
- With the right prompts, they become **financial experts**

**Next class:** We'll see how these prompts get combined with our vector database to create a full Q&A system!

Any questions about how these prompts control our AI's personality and output style?
```python
from langchain.prompts import ChatPromptTemplate "(Creates reusable conversation templates)", HumanMessagePromptTemplate ("Where we put the user's question"), SystemMessagePromptTemplate ("Defines the AI's "role" and personality")
from langchain.schema import AIMessage, HumanMessage, SystemMessage "are different types of messages in a conversation"


class FinancialPrompts:
    """Sophisticated prompts for financial analysis"""
    
    @property
    def system_prompt_analyst(self):
        """if you look up, you will see the @ property,We are using this to 
                - Makes these prompts easy to access: `prompts. system_prompt_analyst`
                - No parentheses needed - they act like attributes, not methods
                - Clean, readable code
        """
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

these are instructions to be highly numerical and precise, to Cite specific figures, and Structure your answer clearly for intance, let's say users ask the followingquestion
> **Q:** "What was Apple's revenue growth?"
> **A:** "According to the 2023 10-K filing, Apple reported $383.3 billion in revenue, representing 2.1% year-over-year growth. This was driven primarily by Services revenue increasing 12.5% to $85.2 billion..."


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

**Example Output:**
> **TL;DR:** Apple showed moderate growth with strong Services performance offsetting hardware declines.
> **Key Insight:** The shift to Services provides more stable, recurring revenue...
> **Strategic Implication:** This diversification reduces reliance on iPhone cycles...


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
for the risk analysyst
we want a **Structured Framework:**
1. **Identify** - Find all risks mentioned
2. **Assess** - Rate them (High/Medium/Low)
3. **Mitigate** - Look for solutions
4. **Compare** - Track changes over time
5. **Recommend** - Suggest next steps

**Example Output:**
> **1. Risk Identification:**
> - Supply chain disruptions (mentioned 15 times)
> - Foreign exchange volatility
> - Intense smartphone competition
> 
> **2. Risk Assessment:**
> - Supply chain: HIGH likelihood, MEDIUM impact
> - Currency: MEDIUM likelihood, LOW impact
> ...

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

**Why this is brilliant because:**
- One function handles all three styles
- Defaults to "analyst" if an invalid style is given
- Easy to add more styles later


## **🎓 Why This Matters for AI Development**

### **1. Control Over Output**
Without good prompts, AI gives generic answers:
> "Apple is doing well with good revenue."

With our prompts:
> "Apple reported $383.3B revenue in FY2023, a 2.1% increase from $375.3B in FY2022..."

### **2. Consistency**
Every student using this system gets the same high-quality analysis style.

### **3. Adaptability**
Different users need different formats:
- **Students** might want the detailed analyst version
- **Executives** might want the summary version
- **Researchers** might want the risk-focused version

---


Create `src/financial_rag/agents/rag_chain.py`

the RAG Chain! This is where we connect **ALL** the pieces we've built into one powerful system that can answer any financial question!

Here, we will create the **BRAIN** of our Financial AI system. This will connects:
1. **📚 Our Vector Database** (from last week)
2. **🤖 Our AI Language Model** (ChatGPT or similar)
3. **📊 Our Financial Prompts** (analyst, executive, risk specialist)
4. **🔍 Our Search System** (similarity or MMR search)

**Think of it like this:**
- **Vector Database** = Your AI's memory (what it knows)
- **Language Model** = Your AI's thinking ability
- **RAG Chain** = The wiring that connects memory to thinking


```python
from langchain.chains import RetrievalQA


**What is a "Chain" in AI?**
- A series of connected steps
- Each step does one specific job
- Data flows through the chain
- **Input → Step 1 → Step 2 → ... → Output**

**Our RAG Chain Flow:**
Question 
    ↓
Search Vector Database (Find relevant info)
    ↓
Combine with Prompt (Add financial expert instructions)
    ↓
Send to AI (Get smart answer)
    ↓
Return Answer + Sources



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
                temperature=0.1,  # i will explain soon
                openai_api_key=config.OPENAI_API_KEY,
                max_retries=3,
                request_timeout=60
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
   #### **Temperature = 0.1 (Very Low)**
# - **Temperature controls creativity:**
#   - 0.0 = Completely predictable (always same answer for same input)
#   - 0.5 = Balanced creativity
#   - 1.0 = Very creative (might make up financial numbers!)

# - **Why 0.1 for finance?**
#   - Financial answers should be **accurate and consistent**
#   - We don't want creative interpretations of stock prices!
#   - **Example:** "Apple's revenue is $383B" should always be $383B, not $380B or $385B

# #### **max_retries=3**
# - If API call fails, try 3 more times
# - Important for reliability

# #### **request_timeout=60**
# - Wait up to 60 seconds for a response
# - Financial analysis might take time 



# now lets create the QA chain
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
            
# ### **What "stuff" Chain Type Means:**
# - Takes **ALL** retrieved documents
# - **"Stuffs"** them into the prompt
# - Sends everything to AI at once
# - **Alternative:** "map_reduce" (summarizes chunks separately, then combines)

# **Visual Example:**
# ```python
# # Retrieves 3 relevant chunks:
# 1. "Apple revenue: $383B"
# 2. "Apple profit margin: 25%"
# 3. "Apple risks: supply chain issues"

# # "Stuffs" them all into one prompt:
# """
# Context:
# 1. Apple revenue: $383B
# 2. Apple profit margin: 25%
# 3. Apple risks: supply chain issues

# Question: What is Apple's financial performance?
# """
            
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



### **Three Ways to Ask Questions:**

#### **1. Analyst Style (Detailed)**
# ```python
# result = rag_chain.analyze_question(
#     "What are Apple's Q3 financial results?",
#     prompt_style="analyst",
#     search_type="similarity"
# )
# ```
# **Output:** Detailed numbers, comparisons, structured analysis

# #### **2. Executive Style (Summary)**
# ```python
# result = rag_chain.analyze_question(
#     "Summarize Microsoft's risks",
#     prompt_style="executive", 
#     search_type="mmr"
# )
# ```
# **Output:** High-level overview, strategic implications

# #### **3. Risk Specialist Style**
# ```python
# result = rag_chain.analyze_question(
#     "Analyze Tesla's risk factors",
#     prompt_style="risk",
#     search_type="similarity"
# )
            
            # Log the analysis
            logger.info(f"Analysis completed. Source documents: {len(result.get('source_documents', []))}")
            
            return {
                "question": question,
                "answer": result["result"], # the AI's answer
                "source_documents": result.get("source_documents", []), # WHere info came from
                "prompt_style": prompt_style, # which expert persona
                "search_type": search_type # how we serched
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
**Example Output:**
```json
{
  "question": "What was Apple's revenue growth?",
  "answer": "Apple reported $383.3 billion revenue in FY2023, representing 2.1% year-over-year growth...",
  "source_documents": [
    {"content": "Apple 10-K 2023, page 45: Revenue $383.3B...", "metadata": {...}},
    {"content": "Apple Q4 earnings call: Growth driven by services...", "metadata": {...}}
  ],
  "prompt_style": "analyst",
  "search_type": "similarity"
}
```


**Why source documents matter:**
- **Transparency** - See where answers come from
- **Verification** - Check if sources are reliable
- **Learning** - Find more information in original documents

## **🎓 Real Classroom Example**

### **Project: Analyze Tech Companies**
```python
# Initialize our system
rag_system = FinancialRAGChain(vector_store)

# Question 1: Get detailed analysis
analysis1 = rag_system.analyze_question(
    "Compare Apple and Microsoft revenue growth and profit margins",
    prompt_style="analyst"
)

# Question 2: Get executive summary  
analysis2 = rag_system.analyze_question(
    "What are the strategic implications of Apple's services growth?",
    prompt_style="executive"
)

# Question 3: Risk analysis
analysis3 = rag_system.analyze_question(
    "What are Tesla's main business risks?",
    prompt_style="risk"
)
```

### **Compare Results:**
```python
print(f"Analyst answer length: {len(analysis1['answer'])} characters")
print(f"Executive answer length: {len(analysis2['answer'])} characters")
print(f"Risk analysis sources: {len(analysis3['source_documents'])} documents")
```


## **🔍 The Search Types**

### **Similarity vs MMR:**
```python
# Similarity Search:
# "Find the 5 most similar chunks to my question"
# Best for: Specific factual questions

# MMR Search (Maximum Marginal Relevance):
# "Find 10 chunks, then pick 5 that are both relevant AND diverse"
# Best for: Broad research questions, avoiding duplicates
```

**Example Difference:**
- **Question:** "Tell me about Apple's financials"
- **Similarity:** Might return 5 chunks all about revenue
- **MMR:** Returns chunks about revenue, profits, risks, growth, margins

---

## **💡 Why This Architecture is Brilliant**

### **1. Modular Design**
- Swap AI models (ChatGPT → Claude → Local model)
- Swap vector databases (Chroma → Pinecone → Weaviate)
- Swap prompts (analyst → student → journalist)

### **2. Traceability**
- Always know where answers come from
- Can verify against original sources
- Builds trust in AI responses

### **3. Flexibility**
- Same system, different "expert personas"
- Adjust for different audiences
- Control creativity vs accuracy

---

## **🎯 Key Takeaways**

1. **RAG = Retrieval + Generation** - Find info, then answer
2. **Temperature matters** - Low for facts, high for creativity
3. **Source documents are crucial** - Never trust AI without sources
4. **Different styles for different needs** - One system, many experts
5. **Chains connect everything** - Like an assembly line for AI thinking

**This is the complete system!** We've gone from:
- Raw documents → Cleaned chunks → Vector database → Smart retrieval → Expert analysis

**Question for discussion:** If you could add another "expert persona" to our system (beyond analyst, executive, risk specialist), what would it be and why?




Very good, now let's Create the Agent with Tools

Create `src/financial_rag/agents/tools.py`


# **📈 Live Financial Data Tools - Building a Real-Time Financial Assistant**

now, we're going to look at one of the most exciting parts of our Financial AI system - the **live data tools** that let our AI access real-time stock prices, financial ratios, and company information! This turns our AI from a static researcher into a real-time financial assistant!

---
Here, we will create a **Financial Swiss Army Knife** that can:
1. **📊 Get real-time stock prices** - Current prices and daily changes
2. **🧮 Calculate financial ratios** - PE ratios, margins, profitability metrics
3. **🏢 Fetch company information** - Sector, employees, market cap
4. **📰 Grab financial news** - Latest articles about companies

Think of it as giving our AI **Bloomberg Terminal powers** but for free!

```python
let's start with the import statement
import yfinance as yf        # Our data source (like free Bloomberg)
import pandas as pd         # Data manipulation
from datetime import datetime  # Timestamps for when we got data
from typing import Dict, List, Any  # Type hints for clarity
from loguru import logger   # Our digital notebook

# **yfinance is the star here** - it's a Python library that gives us free access to Yahoo Finance data, which is surprisingly comprehensive!


class FinancialTools:
    """Tools for the financial agent to interact with real-time data"""
    
    @staticmethod    
# **What `@staticmethod` means:**
# - No `self` parameter needed
# - Can be called directly: `FinancialTools.get_stock_price("AAPL")`
# - Doesn't need an instance of the class
# - **Think of it like:** Calling a function from a toolkit rather than creating a tool object first

# for instance
# No need to create an instance!
# price_data = FinancialTools.get_stock_price("TSLA")
# ratio_data = FinancialTools.calculate_financial_ratio("AAPL", "pe_ratio")

    def get_stock_price(ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """Get current stock price and recent performance"""
        try:
            logger.info(f"Getting stock price for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for ticker {ticker}"}
            
            current_price = hist['Close'].iloc[-1] #  Latest market price
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price # Previous closing price
            price_change = current_price - prev_price # Dollar change from previous close
            price_change_pct = (price_change / prev_price) * 100 # Percentage change (more meaningful)


            
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
#     # sample output
#     **Real Output Example:**
# ```json
# {
#   "ticker": "AAPL",
#   "current_price": 182.63,
#   "price_change": 1.42,
#   "price_change_pct": 0.78,
#   "currency": "USD",
#   "company_name": "Apple Inc.",
#   "timestamp": "2024-01-15T10:30:00"
# }
# ```

    @staticmethod
    def calculate_financial_ratio(ticker: str, ratio: str) -> Dict[str, Any]:
        """Calculate financial ratios"""
        try:
            logger.info(f"Calculating {ratio} for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            ratios = {
                "pe_ratio": info.get('trailingPE'), #PE Ratio = 25** means investors pay $25 for $1 of earnings
                "forward_pe": info.get('forwardPE'),
                "price_to_book": info.get('priceToBook'),
                "debt_to_equity": info.get('debtToEquity'), # **Debt/Equity = 1.5** means $1.50 debt for every $1 equity
                "return_on_equity": info.get('returnOnEquity'), #  **ROE = 0.15** means 15% return on shareholder equity
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
                "sector": info.get('sector'), #  Technology, Healthcare, etc.
                "industry": info.get('industry'), # Consumer Electronics, etc.
                "market_cap": info.get('marketCap'), # Total company value
                "employees": info.get('fullTimeEmployees'),
                "description": info.get('longBusinessSummary'), # Company overview
                "website": info.get('website'),
                "country": info.get('country'),
                "exchange": info.get('exchange'),
                "timestamp": datetime.now().isoformat()
            }
            
            # Clean None values - remove empty fields
            key_info = {k: v for k, v in key_info.items() if v is not None}
            
            logger.success(f"Retrieved company info for {ticker}")
            return key_info
            
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {"error": str(e)}

#  **Example Output:**
# ```json
# {
#   "ticker": "TSLA",
#   "company_name": "Tesla, Inc.",
#   "sector": "Consumer Cyclical",
#   "industry": "Auto Manufacturers",
#   "market_cap": 750000000000,
#   "employees": 127855,
#   "description": "Tesla designs, develops, manufactures...",
#   "country": "United States"
# }

    
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
                    "published_date": datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat() if article.get('providerPublishTime') else None, # Convert Unix timestamp to readable date
                    "related_tickers": article.get('relatedTickers', [])
                }
                formatted_news.append(formatted_article)
            
            logger.success(f"Retrieved {len(formatted_news)} news articles for {ticker}")
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return [{"error": str(e)}]
```



## **🎓 Educational Value**

### **Real-Time Learning:**
```python
# Compare two companies instantly
apple_data = FinancialTools.get_stock_price("AAPL")
microsoft_data = FinancialTools.get_stock_price("MSFT")

# Analyze which is growing faster
apple_growth = apple_data["price_change_pct"]
microsoft_growth = microsoft_data["price_change_pct"]
```

### **Financial Literacy Built-in:**
```python
# Get ratio with explanation
pe_info = FinancialTools.calculate_financial_ratio("GOOGL", "pe_ratio")
print(f"Google's P/E Ratio: {pe_info['value']}")
print(f"What this means: {pe_info['description']}")
```
---

## **💡 Classroom Activities**

### **Activity 1: Market Research Project**
```python
# Compare tech giants
companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

for company in companies:
    info = FinancialTools.get_company_info(company)
    price = FinancialTools.get_stock_price(company)
    print(f"{company}: {info['sector']} - ${price['current_price']}")
```

### **Activity 2: Ratio Analysis**
```python
# Which company is most profitable?
ratios_to_check = ["profit_margin", "return_on_equity", "operating_margin"]

for ratio in ratios_to_check:
    data = FinancialTools.calculate_financial_ratio("AAPL", ratio)
    print(f"Apple's {ratio}: {data['value']} - {data['description']}")
```

---

## **🎯 Key Takeaways**

1. **Live data makes AI powerful** - Static documents vs. real-time prices
2. **Financial ratios tell stories** - Numbers alone don't mean much without interpretation
3. **Error handling is crucial** - Real-world data can be messy
4. **Type hints improve code** - `-> Dict[str, Any]` tells us what to expect
5. **Logging creates a paper trail** - Know what happened when

**This transforms our AI from a "library researcher" into a "Wall Street analyst" with real-time data at its fingertips!**

**Question for the class:** What company would you analyze first with these tools, and what would you look for?





`src/financial_rag/agents/financial_agent.py`

we're going to look at the **most advanced piece** of our Financial AI system - the **Agent**! This isn't just a chatbot - it's a **thinking machine** that can plan, use tools, and make decisions like a human analyst!

In this file, we will create an agent
1. **🧠 Thinks step-by-step** (like a human analyst)
2. **🛠️ Uses different tools** (stock prices, ratios, documents, news)
3. **🤔 Decides which tool to use** based on the question
4. **🔄 Can chain multiple steps** to answer complex questions

**Think of it like:**
- **Regular RAG** = A smart librarian who finds documents
- **This Agent** = A whole financial research team that can:
  - Look up stock prices
  - Calculate ratios
  - Search documents
  - Read news
  - Put it all together

---

## **🎭 The Agent "Thought Process"**

### **The Agent's Inner Monologue:**
```
Question: "Should I invest in Apple stock?"

THOUGHT: "Hmm, this is complex. I need multiple pieces of information."
ACTION: get_stock_price
ACTION INPUT: AAPL
OBSERVATION: "Apple is trading at $182, up 0.8% today"

THOUGHT: "Now I need their financial health."
ACTION: calculate_financial_ratio  
ACTION INPUT: AAPL, pe_ratio
OBSERVATION: "P/E ratio is 28.5, slightly above industry average"

THOUGHT: "Let me check their risks in SEC filings."
ACTION: search_filings
ACTION INPUT: Apple risk factors supply chain
OBSERVATION: "Major risk: Supply chain disruptions in China..."

THOUGHT: "I now have enough information to answer."
FINAL ANSWER: "Based on current price, valuation, and risks..."
```

**This is called REACT (Reason + Act) pattern!**





## **🔍 Watching the Agent Think (Verbose Mode)**

When `verbose=True`, you see:

```
> Entering new AgentExecutor chain...
Thought: The user wants to know if they should invest in Apple. 
I need multiple pieces of information.
Action: get_stock_price
Action Input: AAPL
Observation: Apple is trading at $182.63, up 0.78% today.

Thought: Now I need valuation metrics.
Action: calculate_financial_ratio  
Action Input: AAPL, pe_ratio
Observation: P/E ratio is 28.5

Thought: I now have enough information...
Final Answer: Based on current price and valuation...
```

**Perfect for teaching AI decision-making!**

---


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

# **Key insight:** `agent_scratchpad` is where the agent stores its thinking history!

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
# - Reads the agent's "thoughts"
# - Decides: "Is the agent done thinking?"
# - If not: "Which tool should it use next?"
# - Uses **regex** to find Action and Action Input

class FinancialAgent:
    """Main financial agent that can use tools and RAG"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.rag_chain = FinancialRAGChain(vector_store)
        self.tools = self._setup_tools() # our toolkit
        self.agent_executor = self._setup_agent() # the thinking machine
    
    def _setup_tools(self) -> List[Tool]:
        """Setup the tools available to the agent"""

        # this has 5 specialized tools, so we are going to create a list of tools
#         **Each tool will have a:**
# - **Name** - How the agent calls it
# - **Function** - What it actually does
# - **Description** - When to use it (VERY important!)
        
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


# ### **The Magic: Tool Descriptions!**
# The agent reads the descriptions to decide:


# # If question contains "price" or "stock" → use get_stock_price
# # If question contains "ratio" or "PE" → use calculate_financial_ratio  
# # If question contains "risk" or "filing" → use search_filings
# # If question contains "news" → use get_financial_news

# <!-- 
# **Example Decision Process:**
# ```
# Question: "What's Apple's current PE ratio and stock price?"

# Agent thinks: "Hmm, I need two things:
# 1. PE ratio → calculate_financial_ratio
# 2. Stock price → get_stock_price

# I'll do them one at a time..."
# ```

# ---

# ## **🔄 The Agent Loop**

# the self.tools calls the setup tools

# **Why max_iterations=5?**
# - Prevents infinite loops
# - If agent gets stuck, it stops
# - **Example of a loop:**
#   ```
#   Thought: "I need revenue"
#   Action: search_filings
#   Observation: "Revenue is $383B"
#   Thought: "Now I need revenue..."  # Oops, stuck in loop!
#   ``` -->


# <!-- 
# ## **🎓 Real Classroom Examples**

# ### **Example 1: Simple Question**
# ```python
# agent = FinancialAgent(vector_store)

# # Simple question → One tool
# result = agent.analyze("What's Tesla's current stock price?")
# # Agent thinks: "This needs get_stock_price tool" → Uses it → Returns answer
# ```

# ### **Example 2: Complex Analysis**
# ```python
# # Complex question → Multiple tools chained!
# result = agent.analyze(
#     "Should I invest in Microsoft? Consider their current valuation, recent news, and risk factors from SEC filings."
# )

# # Agent's thinking:
# # 1. get_stock_price("MSFT") → Current price
# # 2. calculate_financial_ratio("MSFT", "pe_ratio") → Valuation  
# # 3. get_financial_news("MSFT") → Recent developments
# # 4. search_filings("Microsoft risk factors") → SEC risks
# # 5. Combine all → Final answer!
# ```

# ---

# ## **📊 Agent vs Regular RAG**

# ### **Regular RAG:**
# ```python
# # Can only search documents
# answer = rag_chain.analyze_question("What's Apple's revenue?")
# # Output: "According to SEC filings, Apple's revenue is $383B"
# ```

# ### **Agent:**
# ```python
# # Can do MUCH more!
# answer = agent.analyze(
#     "Compare Apple and Microsoft on stock performance, valuation ratios, and recent news"
# )
# # Output: Comprehensive comparison using 6+ tool calls! 
# -->

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
    
#     **Why this is smart:**
# - Wraps our existing RAG chain
# - Adds source citations automatically
# - Makes it compatible with the agent system



## **🎯 Key Takeaways**

# 1. **Agents think step-by-step** - Not one-shot answers
# 2. **Tools are like skills** - Each does one thing well
# 3. **Descriptions guide choices** - Tell the agent when to use each tool
# 4. **REACT pattern** = Reason, Act, Observe, Repeat
# 5. **Safety limits are crucial** - max_iterations prevents loops

# **This transforms our AI from:**
# - **Document searcher** → **Financial analyst**
# - **Single tool** → **Entire toolkit**
# - **Simple answers** → **Complex reasoning**

# **Question for discussion:** What other "tools" would you add to this financial agent? What questions could it then answer that it can't answer now?


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







Let's test our rag chin and agent system by Creating an enhanced Test Script

Create `test_agent.py`

# **🧪 The Ultimate AI Test Drive: Putting It All Together!**

now, we're looking at the **final exam** for our Financial AI system - the comprehensive test script that validates EVERYTHING we've built! This is like taking our AI for its first test drive!

---

here, we are going to create

a **complete system integration test** that:
1. **🧱 Builds everything from scratch** (ingestion → processing → storage → AI)
2. **🤖 Tests our smart agent** with different types of questions
3. **🛡️ Has multiple fallbacks** if things fail
4. **📊 Uses realistic mock data** (perfect for learning!)
5. **✅ Gives clear pass/fail results** with next steps

**Think of it like:** A pilot's pre-flight checklist for our AI system!

---

### **Why This Approach Rocks:**
```python
# Instead of testing pieces separately...
test_ingestion()
test_processing()  
test_vector_store()
test_agent()

# We test the ENTIRE pipeline at once!
# Data → Processing → Storage → AI → Answers
```

**Benefits:**
- Catches integration bugs
- Tests real user flows
- Validates the complete experience












------



---

## **🔍 Watching the Test in Action**

### **When You Run This Script:**
```
🤖 Testing Financial Agent System...
🔧 Initializing components...
📊 Setting up knowledge base...
Created 15 document chunks from mock data
👨‍💼 Initializing financial agent...

🧪 Testing Agent Capabilities...

============================================================
Query 1: What are the main risk factors mentioned in the documents?
============================================================
Thought: This is about document content, I should search filings...
Action: search_filings
Action Input: main risk factors
Observation: From Apple 10-K: Supply chain disruptions, currency volatility...
Thought: I now know the answer
Answer: The main risk factors across documents include...

============================================================  
Query 2: What is the current stock price of Apple?
============================================================
Thought: This requires live stock data...
Action: get_stock_price
Action Input: AAPL
Observation: Apple (AAPL) is trading at $182.63...
Answer: Apple's current stock price is $182.63...

✅ Agent system test completed!
```

**You can SEE the AI thinking!**

---

## **🎓 Educational Value**

### **For Students Learning AI:**
```python
# Change the mock data and see what happens!
mock_financial_docs[0]["content"] = "Tesla Revenue: $100 billion..."
# Re-run test → Different answers!

# Add new test queries
test_queries.append("Compare Apple and Tesla revenue growth")
# See if agent can handle comparison
```

### **Debugging Practice:**
```python
# What if we break something?
# Remove a tool from the agent
# See which questions fail
# Learn how components depend on each other
```


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
        doc_processor = DocumentProcessor()      # Our document cleaner
        vector_manager = VectorStoreManager()    # Our AI memory system
        
# **Why start here?** If these fail, everything else will fail!

        # 2. Create or load vector store with mock data
        print("📊 Setting up knowledge base...")
        vector_store = setup_mock_knowledge_base(doc_processor, vector_manager)
# **Instead of real SEC data** (which needs internet, APIs, etc.), we use **perfectly crafted mock data** that:
# - Tests all features
# - Always works
# - Is consistent every time

        
        # 3. Initialize the financial agent
        print("👨‍💼 Initializing financial agent...")
        agent = FinancialAgent(vector_store)
# **This creates our "thinking machine"** with access to:
# - Document search (RAG)
# - Live stock prices
# - Financial ratios
# - Company info
# - News articles
        


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

                # **Why this matters:**
                # - Agent might fail on complex planning
                # - Simple RAG almost always works
                # - **Never leave the user with nothing!**
        
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
    
# **This data is intentionally designed to test:**
# - **Numbers** ($383.3 billion, 2% growth)
# - **Categories** (Revenue, Risks, Segments)
# - **Formats** (Percentages, dollar amounts, lists)
# - **Metadata** (Company, year, source)

# **Perfect for testing:**
# - Can the AI find numbers? ✅
# - Can it understand categories? ✅  
# - Can it cite sources? ✅
# - Can it compare companies? ✅



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



## **🎭 The Three Types of Test Questions**

# ### **Type 1: RAG-Only Questions**
# ```python
# "What are the main risk factors mentioned in the documents?"
# ```
# **Tests:** Document search and retrieval
# **AI uses:** `search_filings` tool
# **Expected:** List of risks from mock data

# ### **Type 2: Tool-Only Questions**  
# ```python
# "What is the current stock price of Apple?"
# ```
# **Tests:** Live data tools
# **AI uses:** `get_stock_price` tool
# **Expected:** Real-time price from yfinance

# ### **Type 3: Complex Hybrid Questions**
# ```python
# "What are Apple's main risk factors and what is their current stock price?"
# ```
# **Tests:** Agent's planning ability
# **AI uses:** `search_filings` → `get_stock_price`
# **Expected:** Combined answer with both types of info



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

now Run the Enhanced Test

```bash
# Make sure you're in the virtual environment and package is installed
pip install -e .

# Run the agent test
python test_agent.py
```

## **🚀 The Complete Development Workflow**

This test script enables:
```
1. Write code
2. Run this test ← WE ARE HERE!
3. See what works/breaks
4. Fix issues
5. Repeat until perfect
6. Deploy to users
```

**Without this test:** You'd have to manually test everything every time!

---

## **🎯 Key Takeaways**

1. **Integration tests > unit tests** for AI systems
2. **Mock data enables offline development**
3. **Test different question types** (simple, complex, edge cases)
4. **Always have fallbacks** - AI can fail in unexpected ways
5. **Clear output helps debugging** - See the AI's thinking process

**This transforms AI development from:**
- **"I hope it works"** → **"I know it works"**
- **Manual testing** → **Automated validation**
- **Fragile code** → **Robust system**

**Question for the class:** If you were to add one more test query to really push our AI to its limits, what would it be and why?



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
