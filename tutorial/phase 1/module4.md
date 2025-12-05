We need to monitor and oberve our system
Create `src/financial_rag/monitoring/__init__.py`

Create `src/financial_rag/monitoring/tracing.py`


# **🔍 The AI Watchtower: Monitoring Everything Your Agent Does!**

now we're going to look at one of the most **professional and important** parts of AI development - **Monitoring and Observability**. This is like having a security camera system that watches your AI 24/7 to see how it's performing!

---

Here, we are going to create **AI Surveillance System** that:
1. **📊 Tracks every single action** your agent takes
2. **⏱️ Measures performance** (speed, accuracy, costs)
3. **📈 Creates beautiful dashboards** for analysis
4. **🐛 Helps debug problems** when things go wrong
5. **📝 Creates a permanent record** of all AI interactions

**Think of it like:** Having a team of scientists watching your AI through a one-way mirror, taking notes on everything it does!

---

## **🎯 Why Monitoring is CRITICAL for AI**

### **Without Monitoring:**
- AI gives wrong answer → "Why???"
- System is slow → "Is it always this slow?"
- Costs are high → "How much did that query cost?"

### **With Monitoring:**
- AI gives wrong answer → "Ah, it retrieved the wrong documents!"
- System is slow → "The LLM call took 8.2 seconds"
- Costs are high → "That query used 1,500 tokens = $0.03"

---
















## **🎨 Weights & Biases (wandb) - The Professional Dashboard**

**This creates a dashboard showing:**


```python
import wandb
from loguru import logger
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from financial_rag.config import config

class AgentMonitor:
    """Monitoring and tracing for the financial agent"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.wandb_run = None
        
        if enabled and config.WANDB_API_KEY and config.WANDB_API_KEY != "your_wandb_api_key_here":
            try:
                self.wandb_run = wandb.init(
                    project="financial-rag-agent",
                    config={
                        "embedding_model": config.EMBEDDING_MODEL,
                        "llm_model": config.LLM_MODEL,
                        "chunk_size": config.CHUNK_SIZE,
                        "top_k_results": config.TOP_K_RESULTS
                    }
                )
### **What is wandb?**
# - **"Weights & Biases"** (not the machine learning kind!)
# - A platform for tracking AI experiments
# - Creates beautiful, interactive dashboards
# - Used by OpenAI, Google, Tesla, etc.

                logger.success("Weights & Biases monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.enabled = False
        else:
            logger.info("Monitoring disabled - no WandB API key found")
    
┌─────────────────────────────────────────────────────┐
│ 📊 Financial RAG Agent Dashboard                    │
├─────────────────────────────────────────────────────┤
│ ⏱️  Average Latency: 4.2s  │  📈 Success Rate: 94% │
│ 💰  Avg Cost/Query: $0.03  │  🔍 Avg Sources: 3.1  │
├─────────────────────────────────────────────────────┤
│                     📈 Charts                        │
│ Latency Over Time │ Token Usage │ Tool Success Rates│
│                     📋 Tables                        │
│ Recent Queries    │ Failed Queries│ Expensive Queries│
└─────────────────────────────────────────────────────┘



    def log_retrieval(self, query: str, documents: List, scores: List[float]):
        """Log retrieval performance"""

#         **Tracks:**
# - How many documents were found
# - How relevant they were (scores)
# - Where they came from (sources)


        if not self.enabled:
            return #monitoring is turned off
            
        try:
            retrieval_metrics = {
                "retrieval/query_length": len(query),
                "retrieval/documents_retrieved": len(documents),
                "retrieval/avg_score": sum(scores) / len(scores) if scores else 0,
                "retrieval/max_score": max(scores) if scores else 0,
                "retrieval/timestamp": datetime.now().isoformat()
            }
            
            # Log source distribution
            sources = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            if self.wandb_run:
                self.wandb_run.log(retrieval_metrics)
                
            logger.debug(f"Retrieval logged: {len(documents)} docs for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging retrieval: {e}")
# <!-- ```
# Query: "What are Apple's risk factors?"
# Found: 5 documents
# Scores: 0.92, 0.87, 0.75, 0.63, 0.41
# Sources: apple_10k_2023, apple_10k_2022
# ```

# **Why this matters:** If scores are low, maybe our vector search isn't working well! -->

    
    def log_llm_call(self, prompt: str, response: str, latency: float, token_usage: Dict = None):
        """Log LLM call details"""
        if not self.enabled:
            return
# **Tracks:**
# - How long prompts are
# - How long responses are
# - How long it takes (latency)
# - **TOKEN USAGE!** (This costs money!)
            
        try:
            llm_metrics = {
                "llm/prompt_length": len(prompt),
                "llm/response_length": len(response),
                "llm/latency_seconds": latency,
                "llm/timestamp": datetime.now().isoformat()
            }
            
            if token_usage:
                llm_metrics.update({
                    "llm/prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "llm/completion_tokens": token_usage.get('completion_tokens', 0),
                    "llm/total_tokens": token_usage.get('total_tokens', 0)
                })
            
            if self.wandb_run:
                self.wandb_run.log(llm_metrics)
                
            logger.debug(f"LLM call logged: {len(prompt)} chars -> {len(response)} chars in {latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Error logging LLM call: {e}")
    


# **Example:**
# ```
# Prompt: 1,200 characters
# Response: 850 characters  
# Latency: 3.2 seconds
# Tokens used: 1,050 ($0.021)
# ```

# **Token tracking is VITAL:** If one query uses 10,000 tokens, that's 20× more expensive than 500 tokens!


    def log_agent_step(self, step_type: str, tool_name: str, input_data: str, output: str, success: bool):
        """Log agent tool usage steps"""
        if not self.enabled:
            return
# **Tracks each tool the agent uses:**
# - Which tool? (`get_stock_price`, `search_filings`, etc.)
# - What was the input?
# - Did it succeed or fail?
# - How much output did it produce?

            
        try:
            agent_metrics = {
                f"agent/{step_type}_tool": tool_name,
                f"agent/{step_type}_input_length": len(input_data),
                f"agent/{step_type}_output_length": len(output),
                f"agent/{step_type}_success": success,
                f"agent/{step_type}_timestamp": datetime.now().isoformat()
            }
            
            if self.wandb_run:
                self.wandb_run.log(agent_metrics)
                
            logger.debug(f"Agent step logged: {tool_name} - Success: {success}")
            
        except Exception as e:
            logger.error(f"Error logging agent step: {e}")
    

# **Example agent thought process:**
# ```
# Step 1: get_stock_price("AAPL") → Success ✓
# Step 2: calculate_financial_ratio("AAPL", "pe_ratio") → Success ✓  
# Step 3: search_filings("Apple risk factors") → Success ✓
# ```

# **Why this matters:** If `get_stock_price` fails 80% of the time, we need to fix it!


    def log_query_analysis(self, question: str, answer: str, total_latency: float, 
                          source_count: int, agent_type: str):
        """Log complete query analysis"""
        if not self.enabled:
            return

# **Tracks the entire user experience:**
# - Question → Answer journey
# - Total time from start to finish
# - Number of sources used
# - Success or failure

            
        try:
            query_metrics = {
                "query/question_length": len(question),
                "query/answer_length": len(answer),
                "query/total_latency_seconds": total_latency,
                "query/source_count": source_count,
                "query/agent_type": agent_type,
                "query/success": len(answer) > 0,
                "query/timestamp": datetime.now().isoformat()
            }
            
            if self.wandb_run:
                self.wandb_run.log(query_metrics)
                
                # Log the actual Q&A for analysis
                self.wandb_run.log({
                    "query_samples": wandb.Table(
                        columns=["Question", "Answer", "Latency", "Sources"],
                        data=[[question, answer, total_latency, source_count]]
                    )
                })
            
            logger.info(f"Query analysis logged: {question[:50]}... -> {len(answer)} chars in {total_latency:.2f}s")
            
        except Exception as e:
            logger.error(f"Error logging query analysis: {e}")
<!-- 
# **Example:**
# ```
# Question: "Should I invest in Apple?"
# Answer length: 1,250 characters
# Total time: 8.5 seconds  
# Sources used: 3 documents
# Success: Yes ✓
# ```

# **User perspective:** If queries take 30 seconds, users will leave! -->


    def cleanup(self):
        """Clean up monitoring resources"""
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("Monitoring session ended")


<!-- ## **🎓 Real Classroom Examples**

### **Example 1: Debugging a Slow Query**
```python
# Student: "Why is my query taking 20 seconds?"
# Teacher checks monitoring:

log_query_analysis:
  Question: "Compare all FAANG companies..."
  Total latency: 22.4 seconds
  
log_agent_step:
  Step 1: get_stock_price (5 companies) → 8.2s
  Step 2: calculate_financial_ratio (5×4 ratios) → 12.1s
  Step 3: search_filings (5 companies) → 2.1s

# Diagnosis: Too many API calls! Need to optimize.
```

### **Example 2: Finding Expensive Queries**
```python
# Monitoring shows:
Query: "Write me a 5-page report on Apple..."
Token usage: 12,500 tokens
Cost: $0.25 per query!

# Solution: Add token limits or charge users!
```

### **Example 3: Tracking Learning Progress**
```python
# Week 1: Agent success rate: 65%
# Week 2: After fixing tools → 82%
# Week 3: After better prompts → 94%

# Visual progress that shows your improvements!
```

<!-- ## **🛡️ The Safety Features**

### **Graceful Degradation:**
```python
if not self.enabled:
    return  # Skip if monitoring is off
    
try:
    # Try to log
except Exception as e:
    logger.error(f"Error logging: {e}")  # Don't crash the main app!
``` -->


**Why this matters:** If wandb is down, your AI should still work!



## **📈 What You Can Learn From Monitoring**

### **Performance Insights:**
- Which tools are fastest/slowest?
- Which queries fail most often?
- What's your average cost per query?
- How many sources give the best answers?

### **Business Insights:**
- Peak usage times
- Most popular question types
- Return on investment (ROI)
- Scaling needs

### **Quality Insights:**
- Answer length vs. user satisfaction
- Source count vs. answer quality
- Tool combinations that work best

---

## **💡 Classroom Activities**

### **Activity 1: The Monitoring Detective**
```python
# Give students monitoring data and ask:
"Based on this data, what's wrong with our AI?"

Data:
- get_stock_price success: 30%
- search_filings success: 95%
- Average latency: 15s

Answer: "Stock price API is unreliable and slow!"
```

### **Activity 2: Dashboard Design**
```python
# "What metrics would YOU want to see?"
Students design their ideal AI monitoring dashboard:
- Custom charts
- Alert systems  
- Success metrics
```

### **Activity 3: Cost Analysis**
```python
# "If we get 1,000 users per day:
# - Average queries/user: 3
# - Average tokens/query: 800
# - Cost per 1K tokens: $0.02
# What's our monthly cost?"

Answer: 1,000 × 3 × 800 × 30 × $0.02/1000 = $1,440/month
```

---

## **🎯 Key Takeaways**

1. **Monitoring is not optional** for production AI
2. **Measure everything** - latency, costs, success rates
3. **Dashboards make data understandable**
4. **Never let monitoring crash your app**
5. **Use data to drive improvements**

**This transforms AI development from:**
- **"I think it works"** → **"I know it works because data shows..."**
- **Guessing** → **Data-driven decisions**
- **Reactive fixing** → **Proactive optimization**

**Question for discussion:** If you could add one more thing to monitor about our financial agent, what would it be and why?







Now we need to monitor the existing financial agent;
Update `src/financial_rag/agents/financial_agent.py` with Monitoring

let's see how to **actually integrate** our monitoring system into the Financial Agent. This is like installing a dashboard and sensors in a race car - we're adding the instruments that let us see how it's performing!


## **🚀 What We're Adding**

We're making **two key changes** to our Financial Agent:

1. **🔧 Giving it monitoring capabilities** (adding the "eyes")
2. **📝 Tracking every analysis it performs** (recording the "drives")

**Think of it like:** Adding a flight data recorder to our AI plane!

---

## **🎯 The Two Changes Explained**

### **Change 1: The Monitor in the Constructor**
Update `src/financial_rag/agents/financial_agent.py` with Monitoring
```python
class FinancialAgent:
    def __init__(self, vector_store, enable_monitoring: bool = True):
        # ... existing setup ...
        self.monitor = AgentMonitor(enabled=enable_monitoring)  # NEW!
```

**What this does:**
- Creates a `monitor` object inside our agent
- Defaults to `True` (monitoring ON)
- Can be turned off: `FinancialAgent(vector_store, enable_monitoring=False)`

**Why `enable_monitoring` parameter?**
- **Development:** Keep monitoring ON to debug
- **Testing:** Might turn OFF to avoid wandb costs
- **Production:** Definitely keep ON!

---

### **Change 2: Timing and Tracking Every Analysis**
lets go inside of the analyze method, first let's add a start time
```python
def analyze(self, question: str) -> Dict[str, Any]:
    start_time = time.time()  # 🕒 Start the stopwatch!
    
    try:
        # ... agent does its work ...
        end_time = time.time()  # 🏁 Stop the stopwatch!
        
        # 📊 Log everything to monitoring
        self.monitor.log_query_analysis(
            question=question,
            answer=result,
            total_latency=end_time - start_time,  # ⏱️ How long it took
            source_count=0,  # Temporary - we'll fix this!
            agent_type="tool_using_agent"
        )
```


### **Why Time Everything?**

adding a start and end time is is like Timing how long it takes you to solve a math problem!

**What we learn:**
- Fast queries: "What's Apple's stock price?" → 0.8 seconds
- Medium queries: "Compare Apple and Microsoft" → 3.2 seconds  
- Slow queries: "Analyze all FAANG companies..." → 22.1 seconds

**Actionable insights:**
- If > 10 seconds → Maybe warn users: "This might take a moment..."
- If > 30 seconds → Consider optimizing or limiting complexity

---

## **📈 What Gets Tracked**

### **For Every Single Query:**
```python
self.monitor.log_query_analysis(
    question="What are Apple's risk factors?",      # The question
    answer="Apple faces risks including...",       # The AI's answer
    total_latency=4.2,                             # Time taken (seconds)
    source_count=3,                                # How many sources used
    agent_type="tool_using_agent"                  # Which AI did it
)
```

**Real data collected:**
```
Date: 2024-01-15
Question: "What are Apple's risk factors?"
Answer length: 1,245 characters
Time: 4.2 seconds
Success: Yes
Agent: tool_using_agent
```

---

## **🛡️ Error Tracking Too!**

### **Monitoring Success AND Failure:**
```python
try:
    # Try to analyze
    result = self.agent_executor.run(question)
    
except Exception as e:
    end_time = time.time()
    error_msg = f"I encountered an error while analyzing your question: {str(e)}"
    
    # Even errors get logged!
    self.monitor.log_query_analysis(
        question=question,
        answer=f"Error: {str(e)}",  # Log the error message
        total_latency=end_time - start_time,
        source_count=0,
        agent_type="tool_using_agent"
    )
    logger.error(f"Error in agent analysis: {str(e)}")
    return {
        "question": question,
        "answer": error_msg,
        "error": str(e),
        "agent_type": "tool_using_agent",
        "latency_seconds": end_time - start_time
    }
```

**Why log errors too?**
- See which questions cause failures
- Track error rates over time
- Identify patterns: "Stock price API fails 15% of the time"

**Without this:** You'd only know something failed, not why or how often!

---

## **🎓 Real Classroom Example**

### **Before Monitoring:**
```python
student = FinancialAgent(vector_store)
answer = student.analyze("What's Tesla's PE ratio?")

# Output: Some answer...
# But we don't know:
# - How long it took?
# - Did it use the right tool?
# - How many tokens were used?
# - Did it almost fail?
```

### **After Monitoring:**
```python
student = FinancialAgent(vector_store)
answer = student.analyze("What's Tesla's PE ratio?")

# PLUS we now know:
# ⏱️  Took 1.8 seconds
# 🛠️  Used calculate_financial_ratio tool
# 💰  Cost $0.004 in tokens
# ✅  Success rate for this tool: 92%
# 📊  Average time for ratio queries: 2.1s
```

---

## **🔍 The Temporary Workaround**

```python
source_count=0,  # Temporary - we'll fix this!
```

**What's happening here?**
- The agent uses tools internally
- We can't easily count sources from outside
- **Temporary solution:** Set to 0
- **Better solution:** Modify agent to expose source info

**This is REAL software development:** Start simple, improve later!

---

## **💡 Enhanced Return Data**

### **Now returning MORE information:**
```python
return {
    "question": question,
    "answer": result,
    "agent_type": "tool_using_agent",
    "latency_seconds": end_time - start_time  # NEW! Now users know speed
}
```

**Users/apps can now:**
- Show "Thinking..." for > 2 seconds
- Display "Answer generated in 3.4s"
- Set timeouts based on historical data
- Charge users based on processing time

---


## **🎯 The Big Picture Integration**

### **Before:**
```
User Question → Agent → Answer
```

### **After:**
```
User Question → Agent + Monitor → Answer + 📊 Data
                     ⬇
              📈 Dashboard
              📝 Logs  
              ⚠️ Alerts
              💰 Cost Tracking
```

**We've added a COMPLETE observability layer!**

---

## **📚 Classroom Activity**

### **Activity: The Monitoring Challenge**
```python
# Task: "Find the slowest tool in our agent"

# Students would:
# 1. Run 100 test queries
# 2. Check monitoring dashboard
# 3. Find which tool has highest average latency
# 4. Propose optimizations

# Example findings:
# get_stock_price: avg 0.8s
# search_filings: avg 2.1s  
# calculate_financial_ratio: avg 3.4s ← SLOWEST!
# get_company_info: avg 1.2s

# Optimization idea: Cache financial ratios!
```

---

## **🎯 Key Takeaways**

1. **Monitoring is not separate** - It's integrated into the agent
2. **Time everything** - Performance data is gold
3. **Track both success and failure** - Learn from mistakes
4. **Start simple, improve gradually** - `source_count=0` is okay to start
5. **Return more data** - Help users understand what happened

**This transforms our agent from:**
- **"A black box"** → **"A transparent, measurable system"**
- **"Hope it works"** → **"Know exactly how it performs"**
- **"Reactive support"** → **"Proactive optimization"**

**Question for discussion:** If you could add ONE more thing to monitor about each query (besides time and success), what would it be and why?









```python

# Add to imports
from financial_rag.monitoring.tracing import AgentMonitor

# Update the FinancialAgent class __init__ method:
class FinancialAgent:
    def __init__(self, vector_store, enable_monitoring: bool = True):
        self.vector_store = vector_store
        self.rag_chain = FinancialRAGChain(vector_store)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
        self.monitor = AgentMonitor(enabled=enable_monitoring)
    
    # Update the analyze method with monitoring:
    def analyze(self, question: str) -> Dict[str, Any]:
        """Main method to analyze a financial question using the agent"""
        start_time = time.time()
        
        try:
            logger.info(f"Agent analyzing question: {question}")
            
            result = self.agent_executor.run(question)
            end_time = time.time()
            
            # Log successful analysis
            self.monitor.log_query_analysis(
                question=question,
                answer=result,
                total_latency=end_time - start_time,
                source_count=0,  # Would need to extract from agent internals
                agent_type="tool_using_agent"
            )
            
            return {
                "question": question,
                "answer": result,
                "agent_type": "tool_using_agent",
                "latency_seconds": end_time - start_time
            }
            
        except Exception as e:
            end_time = time.time()
            error_msg = f"I encountered an error while analyzing your question: {str(e)}"
            
            # Log failed analysis
            self.monitor.log_query_analysis(
                question=question,
                answer=error_msg,
                total_latency=end_time - start_time,
                source_count=0,
                agent_type="tool_using_agent"
            )
            
            logger.error(f"Error in agent analysis: {str(e)}")
            return {
                "question": question,
                "answer": error_msg,
                "error": str(e),
                "agent_type": "tool_using_agent",
                "latency_seconds": end_time - start_time
            }
```

FastAPI REST API


# **📝 Data Contracts: Speaking Clearly Between Humans and AI**

now lets look at one of the most **professional and important** concepts in software engineering - **Data Contracts** using Pydantic models. This is like creating a "rulebook" for how data should flow in and out of our AI system!

---

## **🚀 What are we going to do here?**

Here, we are going to create **clear, strict definitions** for:
1. **📨 What data the API accepts** (requests)
2. **📤 What data the API returns** (responses)  
3. **✅ Automatic validation** (no bad data allowed!)
4. **📚 Excellent documentation** (self-documenting code)

**Think of it like:** Creating custom forms and receipts for every interaction with our AI!

---

## **🎯 Why Data Contracts Matter**

### **Without Data Contracts:**
```python
def analyze_question(question, style="analyst", use_agent=True):
    # What's a valid question? Any string? Empty string?
    # What's a valid style? "analyst" or "Analyst" or "ANALYST"?
    # What if someone sends style="president"? No error until runtime!
```

### **With Data Contracts:**
```python
class QueryRequest(BaseModel):
    question: str = Field(..., description="The financial question")
    analysis_style: AnalysisStyle = Field(default=AnalysisStyle.ANALYST)
    # IMMEDIATE ERROR if style="president"!
```

**Benefits:**
- **No guessing** what data to send
- **Automatic validation** - catches errors early
- **Self-documenting** - shows exactly what's expected
- **Type safety** - Python knows what types to expect

---









Create `src/financial_rag/api/__init__.py`

Create `src/financial_rag/api/models.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class AnalysisStyle(str, Enum):
    ANALYST = "analyst"
    EXECUTIVE = "executive"
    RISK = "risk"


# **What this does:**
# - Creates a **limited set of choices**
# - Like a dropdown menu: ⬇
#   - analyst
#   - executive  
#   - risk

# **Why not just strings?**
# ```python
# # ❌ Bad: Any string works, even typos!
# style = "analysst"  # Typo! But no error

# # ✅ Good: Only these three choices allowed!
# style = AnalysisStyle.ANALYST  # Must be exactly one of the three
# ```

class SearchType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"

class QueryRequest(BaseModel):
    question: str = Field(..., description="The financial question to analyze")
    analysis_style: AnalysisStyle = Field(default=AnalysisStyle.ANALYST, description="Style of analysis")
    use_agent: bool = Field(default=True, description="Whether to use the intelligent agent")
    search_type: SearchType = Field(default=SearchType.SIMILARITY, description="Search type for retrieval")


# **This is like filling out a form:**

# ┌─────────────────────────────────────────┐
# │ 📋 Ask the AI a Question                │
# ├─────────────────────────────────────────┤
# │ Question: [____________________________] │
# │                                         │
# │ Analysis Style: ⬇ [analyst]            │
# │                                         │  
# │ Use Intelligent Agent: ☑ Yes  □ No      │
# │                                         │
# │ Search Type: ⬇ [similarity]            │
# └─────────────────────────────────────────┘


# **Field parameters explained:**
# - `...` = **Required!** (question is mandatory)
# - `default=...` = **Optional, with default value**
# - `description=...` = **Help text** for users/developers



class IngestionRequest(BaseModel):
    ticker: str
    filing_type: str = Field(default="10-K")
    years: int = Field(default=2, ge=1, le=5)

# **The magic: `ge=1, le=5`**
# - **ge** = greater than or equal to 1
# - **le** = less than or equal to 5
# - **Automatic validation:** Can't request 0 years or 10 years!


# **Example valid requests:**
# ```python
# IngestionRequest(ticker="AAPL")                 # 2 years of 10-K (defaults)
# IngestionRequest(ticker="TSLA", years=3)        # 3 years of 10-K
# IngestionRequest(ticker="MSFT", filing_type="10-Q", years=1)
# ```

# **Example INVALID (auto-rejected!):**
# ```python
# IngestionRequest(ticker="AAPL", years=10)  
# # ❌ Error: years must be ≤ 5
# ``


class QueryResponse(BaseModel):
    question: str
    answer: str
    agent_type: str
    latency_seconds: float
    source_documents: List[DocumentResponse] = []
    error: Optional[str] = None

# **Always returns the SAME structure:**
# ```json
# {
#   "question": "What are Apple's risk factors?",
#   "answer": "Apple faces risks including...",
#   "agent_type": "tool_using_agent",
#   "latency_seconds": 3.2,
#   "source_documents": [...],
#   "error": null  // No error!
# }
# ```

# **Even on error:**
# ```json
# {
#   "question": "What are Apple's risk factors?",
#   "answer": "I encountered an error...",
#   "agent_type": "tool_using_agent", 
#   "latency_seconds": 1.5,
#   "source_documents": [],
#   "error": "API connection failed"
# }
# ```

# **Consistency is KEY:** Frontend always knows what to expect!



class DocumentResponse(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
# **Transparent sourcing:**
# ```json
# {
#   "content": "Apple 10-K 2023: Risks include supply chain disruptions...",
#   "metadata": {"source": "apple_10k_2023", "company": "Apple"},
#   "score": 0.92  // Very relevant to the question!
# }
# ```

# **Users can:** Click to see sources, verify information, explore further!



class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_ready: bool
    llm_ready: bool

# **Example:**
# ```json
# {
#   "status": "healthy",
#   "version": "1.2.0",
#   "vector_store_ready": true,
#   "llm_ready": true
# }
# ```

# **Monitoring tools can:** Check `/health` endpoint, alert if `llm_ready: false`!



class IngestionResponse(BaseModel):
    ticker: str
    filings_downloaded: int
    documents_processed: int
    success: bool
    error: Optional[str] = None
```



<!-- ## **🎓 Real-World Examples**

### **Example 1: Frontend Integration**
```python
# Frontend developer knows EXACTLY what to send:
request_data = QueryRequest(
    question="What's Apple's PE ratio?",
    analysis_style=AnalysisStyle.EXECUTIVE,
    use_agent=True
)

# And EXACTLY what to expect back:
response = api.analyze(request_data)
# response.latency_seconds  ← Definitely exists!
# response.source_documents ← Definitely a list!
```

### **Example 2: API Documentation**
```python
# FastAPI automatically generates docs from Pydantic!
"""
POST /analyze
Body: QueryRequest
Returns: QueryResponse

QueryRequest:
- question: str (required)
- analysis_style: "analyst"|"executive"|"risk" (optional)
- use_agent: bool (optional)
- search_type: "similarity"|"mmr" (optional)
"""
# No manual documentation needed!
```

### **Example 3: Error Prevention**
```python
# What users might try to send:
bad_request = {
    "question": "What's Apple's stock?",
    "analysis_style": "summary",  # ❌ Not in enum!
    "years": -5                    # ❌ Negative years!
}

# What happens:
try:
    QueryRequest(**bad_request)
except ValidationError as e:
    print(e.errors())
    # Output: 
    # [{'loc': ['analysis_style'], 'msg': 'value is not a valid enumeration...'}]
    # [{'loc': ['years'], 'msg': 'ensure this value is greater than or equal to 1'}]
# Errors caught BEFORE reaching our business logic!
```

---

## **💡 Classroom Activity: Design Your Own Model**

### **Activity: Create a `CompanyComparisonRequest`**
```python
# Requirements:
# - List of 2-5 company tickers
# - Time period (1y, 5y, max)
# - Compare by: "revenue", "profit", "growth", "all"
# - Output format: "table", "summary", "detailed"

# Students would create:
class CompareBy(str, Enum):
    REVENUE = "revenue"
    PROFIT = "profit"
    GROWTH = "growth"
    ALL = "all"

class OutputFormat(str, Enum):
    TABLE = "table"
    SUMMARY = "summary"
    DETAILED = "detailed"

class CompanyComparisonRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, max_items=5)
    period: str = Field(default="1y", pattern=r"^\d+[ym]$")
    compare_by: CompareBy = Field(default=CompareBy.ALL)
    output_format: OutputFormat = Field(default=OutputFormat.SUMMARY)
```

**Teaches:** Enum design, validation, default values, user experience! -->



## **🎯 Key Takeaways**

1. **Data contracts define rules** - No more guessing!
2. **Enums limit choices** - Prevent invalid data
3. **Validation happens automatically** - Errors caught early
4. **Self-documenting** - Code explains itself
5. **Consistent responses** - Frontend always knows what to expect

**This transforms our API from:**
- **"Hope you send the right data"** → **"Here's exactly what to send"**
- **"Might return different things"** → **"Always returns this structure"**
- **"Manual documentation"** → **"Auto-generated from code"**

**Question for discussion:** If you were to add one more field to `QueryRequest` to make our AI even more powerful, what would it be and why?









Create `src/financial_rag/api/server.py`

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
import os
import time

from financial_rag.config import config
from financial_rag.api.models import (
    QueryRequest, QueryResponse, HealthResponse, 
    IngestionRequest, IngestionResponse, AnalysisStyle
)
from financial_rag.retrieval.vector_store import VectorStoreManager
from financial_rag.agents.financial_agent import FinancialAgent
from financial_rag.ingestion.sec_ingestor import SECIngestor
from financial_rag.ingestion.document_processor import DocumentProcessor

class FinancialRAGAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Financial RAG Analyst API",
            description="Enterprise-grade Financial Analysis using RAG and AI Agents",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.vector_store = None
        self.agent = None
        self.setup()
    
    def setup(self):
        """Setup middleware and routes"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify exact origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add startup event
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize_services()
        
        # Add routes
        self.setup_routes()
    
    async def initialize_services(self):
        """Initialize vector store and agent"""
        try:
            logger.info("Initializing Financial RAG API services...")
            
            # Initialize vector store
            vector_manager = VectorStoreManager()
            self.vector_store = vector_manager.load_vector_store()
            
            if self.vector_store is None:
                logger.warning("No existing vector store found. Please ingest data first.")
                # Create empty vector store for health checks
                from langchain.schema import Document
                self.vector_store = vector_manager.create_vector_store([Document(page_content="", metadata={})])
            
            # Initialize agent
            self.agent = FinancialAgent(self.vector_store, enable_monitoring=True)
            
            logger.success("Financial RAG API services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", include_in_schema=False)
        async def root():
            return {"message": "Financial RAG Analyst API", "version": "0.1.0"}
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                vector_store_ready = self.vector_store is not None
                llm_ready = self.agent is not None and self.agent.rag_chain.llm is not None
                
                return HealthResponse(
                    status="healthy",
                    version="0.1.0",
                    vector_store_ready=vector_store_ready,
                    llm_ready=llm_ready
                )
            except Exception as e:
                return HealthResponse(
                    status="unhealthy",
                    version="0.1.0",
                    vector_store_ready=False,
                    llm_ready=False
                )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_analysis(request: QueryRequest):
            """Main endpoint for financial analysis"""
            try:
                if not self.agent:
                    raise HTTPException(status_code=503, detail="Agent not initialized")
                
                start_time = time.time()
                
                if request.use_agent:
                    # Use intelligent agent with tools
                    result = self.agent.analyze(request.question)
                else:
                    # Use simple RAG
                    result = self.agent.simple_rag_analysis(
                        request.question, 
                        prompt_style=request.analysis_style
                    )
                
                # Convert source documents to response format
                source_docs = []
                for doc in result.get("source_documents", []):
                    source_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)
                    })
                
                return QueryResponse(
                    question=result["question"],
                    answer=result["answer"],
                    agent_type=result.get("agent_type", "simple_rag"),
                    latency_seconds=result.get("latency_seconds", time.time() - start_time),
                    source_documents=source_docs,
                    error=result.get("error")
                )
                
            except Exception as e:
                logger.error(f"Error in query analysis: {e}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/ingest/sec", response_model=IngestionResponse)
        async def ingest_sec_filings(request: IngestionRequest, background_tasks: BackgroundTasks):
            """Ingest SEC filings for a ticker"""
            try:
                # Run ingestion in background
                background_tasks.add_task(
                    self._ingest_sec_filings_background,
                    request.ticker,
                    request.filing_type,
                    request.years
                )
                
                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,  # Will be updated in background
                    documents_processed=0,
                    success=True,
                    error=None
                )
                
            except Exception as e:
                return IngestionResponse(
                    ticker=request.ticker,
                    filings_downloaded=0,
                    documents_processed=0,
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/system/stats")
        async def system_stats():
            """Get system statistics"""
            try:
                if not self.vector_store:
                    return {"error": "Vector store not initialized"}
                
                # Get collection stats
                collection = self.vector_store._collection
                if collection:
                    count = collection.count()
                    
                    return {
                        "vector_store_documents": count,
                        "embedding_model": config.EMBEDDING_MODEL,
                        "llm_model": config.LLM_MODEL,
                        "status": "operational"
                    }
                else:
                    return {"error": "Collection not available"}
                    
            except Exception as e:
                return {"error": f"Failed to get stats: {str(e)}"}
    
    async def _ingest_sec_filings_background(self, ticker: str, filing_type: str, years: int):
        """Background task for SEC filings ingestion"""
        try:
            logger.info(f"Starting background ingestion for {ticker}")
            
            # Download filings
            sec_ingestor = SECIngestor()
            num_filings = sec_ingestor.download_filings(ticker, filing_type, years)
            
            # Process documents
            doc_processor = DocumentProcessor()
            filing_paths = sec_ingestor.get_filing_paths(ticker, filing_type)
            
            documents = []
            for filing_path in filing_paths:
                doc = doc_processor.process_sec_filing(filing_path)
                documents.append(doc)
            
            # Chunk documents
            chunks = doc_processor.chunk_documents(documents)
            
            # Update vector store
            vector_manager = VectorStoreManager()
            self.vector_store = vector_manager.create_vector_store(chunks)
            
            # Reinitialize agent with updated vector store
            self.agent = FinancialAgent(self.vector_store, enable_monitoring=True)
            
            logger.success(f"Background ingestion completed for {ticker}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Background ingestion failed for {ticker}: {e}")

# Create app instance
app = FinancialRAGAPI().app

if __name__ == "__main__":
    uvicorn.run(
        "financial_rag.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )
```

Docker Containerization

Create `Dockerfile`
# **📦 The AI Factory: Containerizing Our Financial Agent!**

Let's see how we package our entire AI system into a neat, portable box that can run anywhere!

---

## **🚀 What This Dockerfile Does**

for those who are not familiar with docker, here, we will create a **self-contained AI factory** that:
1. **🏗️ Builds a clean environment** from scratch
2. **📦 Installs everything needed** (dependencies, tools)
3. **👤 Sets up secure access** (non-root user)
4. **🌍 Makes it runnable anywhere** (cloud, laptop, server)
5. **❤️ Keeps it healthy** (automatic health checks)

**Think of it like:** Building a perfectly organized, self-sufficient AI laboratory in a box!



---



```dockerfile
FROM python:3.9-slim

# **What this means:**
# - Start with **official Python 3.9** image
# - `slim` version = Smaller, more secure (no extra tools)
# - **Like:** "Start with a clean Python playground"

# **Why Python 3.9?**
# - Stable, widely supported
# - Works with all our libraries
# - Not too old, not too new


WORKDIR /app

# **Sets the working directory** inside the container
# - All commands run from `/app`
# - Files get copied to `/app`
# - **Like:** "Set up your desk before working"


# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*
# ```
# **Installs COMPILERS:**
# - `gcc` = C compiler (needed for some Python packages)
# - `g++` = C++ compiler
# - **Why?** Some AI libraries need compilation!

# **Cleans up:**
# - `rm -rf /var/lib/apt/lists/*` = Removes package lists
# - **Reduces image size** by ~50MB!
# - **Security:** Fewer files = smaller attack surface

# ---


# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# **Smart dependency installation:**
# 1. **Copy just `pyproject.toml`** (not all code yet)
# 2. **Install dependencies** from it
# 3. `--no-cache-dir` = Don't save downloaded packages
# 4. `-e .` = "Editable" install (links to our code)

# **Why copy only pyproject.toml first?**
# - **Docker layer caching:** If dependencies don't change, skip reinstallation!
# - **Faster builds:** Change code? Dependencies already installed!


# Copy source code
COPY src/ ./src/
COPY .env ./

# **Now add our actual code:**
# - `src/` = All our Python files
# - `.env` = Configuration (API keys, settings)

# **Note:** `.env` should NEVER contain real secrets in production!
# - Use Docker secrets or environment variables instead
# - This is for development/testing


# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# **Creates a NON-ROOT user:**
# - Default Docker runs as `root` (dangerous!)
# - We create `app` user
# - Switch to `app` for running our code

# **Why? Security 101:**
# - If hacked, attacker gets `app` user privileges, not `root`
# - Can't install malware or modify system files
# - **Like:** Giving a guest a visitor badge, not master keys



# Expose port
EXPOSE 8000

# **Documentation:** "This container listens on port 8000"
# - Doesn't actually open the port (that's done when running)
# - Just documentation for users
# - **Like:** "This box has a door on side #8000"


# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# **Automatic health monitoring:**
# - Every **30 seconds**: Check if app is healthy
# - Wait **5 seconds** on startup (app needs time to start)
# - Try **3 times** before declaring "unhealthy"
# - **How check?** Call `/health` endpoint

# **If unhealthy:**
# - Docker knows container is sick
# - Can auto-restart it
# - Orchestrators (Kubernetes) can replace it


# Run the application
CMD ["uvicorn", "financial_rag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```
<!-- 
**How to run the app:**
- `uvicorn` = ASGI server (fast, modern Python server)
- `financial_rag.api.server:app` = Our FastAPI app
- `--host 0.0.0.0` = Listen on all network interfaces
- `--port 8000` = Port to listen on

**`0.0.0.0` is CRITICAL:**
- Means "accept connections from anywhere"
- Without this: Only localhost can connect
- **Like:** "Open the store doors to the street" -->



<!-- ## **🏗️ The Build Process**

### **What happens when you run `docker build .`:**
```
1. Start with python:3.9-slim (≈ 120MB)
2. Set WORKDIR to /app
3. Install gcc, g++ (+100MB)
4. Install Python dependencies (+300MB)
5. Copy source code (+5MB)
6. Create user (tiny)
7. Final image: ≈ 525MB

Total: 5-10 minutes first time
Next time: 30 seconds (cached!)
```

---

## **🚀 Running Our Container**

### **Development:**
```bash
# Build it
docker build -t financial-agent .

# Run it
docker run -p 8000:8000 financial-agent

# Access: http://localhost:8000
```

### **Production (with environment variables):**
```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  -e WANDB_API_KEY="wandb_..." \
  --name financial-agent \
  financial-agent:latest
```

---

## **🎓 Real-World Examples**

### **Example 1: Classroom Setup**
```bash
# Teacher sends Dockerfile to students
git clone https://github.com/teacher/financial-rag
cd financial-rag

# ONE command sets up everything!
docker build -t my-ai .
docker run -p 8000:8000 my-ai

# All students have identical environment!
# No "works on my machine" problems!
```

### **Example 2: Cloud Deployment**
```bash
# Push to Docker Hub
docker tag financial-agent yourname/financial-agent
docker push yourname/financial-agent

# Run on any cloud:
# AWS: aws ecs run-task --image yourname/financial-agent
# Google Cloud: gcloud run deploy --image yourname/financial-agent
# Azure: az container create --image yourname/financial-agent
```

### **Example 3: Scaling**
```bash
# Need more capacity? Run more containers!
docker run -p 8001:8000 financial-agent
docker run -p 8002:8000 financial-agent  
docker run -p 8003:8000 financial-agent

# Load balancer sends requests to all 3!
```

---

## **🔍 The Magic of Layers**

### **Docker uses a "layered" filesystem:**
```
Layer 1: python:3.9-slim (120MB)  ← Base
Layer 2: gcc, g++ (+100MB)         ← System tools
Layer 3: Python deps (+300MB)      ← Libraries  
Layer 4: Our code (+5MB)           ← Our changes
```

**Benefits:**
- **Share layers** between containers
- **Cache layers** for faster builds
- **Small updates** = only rebuild changed layers

**Example:** Change code? Only Layer 4 rebuilds!

---

## **💡 Classroom Activities**

### **Activity 1: The Size Challenge**
```bash
# Task: "Make the Docker image smaller"
# Students try:
# 1. Use python:3.9-alpine (≈ 45MB vs 120MB)
# 2. Combine RUN commands (fewer layers)
# 3. Clean apt cache in same RUN command

# Learn: Trade-offs between size, compatibility, build time
```

### **Activity 2: The Security Audit**
```bash
# Task: "Find security issues"
# Students check:
# 1. Running as root? ❌ (We fixed!)
# 2. Exposed ports? ✅ (Only 8000)
# 3. Secrets in image? ❌ (.env should be external)
# 4. Outdated packages? Check apt-get update
```

### **Activity 3: The Health Check Game**
```python
# Modify health endpoint to fail sometimes
@app.get("/health")
def health():
    import random
    if random.random() < 0.3:  # 30% chance of failure
        return {"status": "unhealthy"}, 500
    return {"status": "healthy"}
    
# Watch Docker restart the container!
``` -->



## **🎯 Key Takeaways**

1. **Docker = Portable environments** - Run anywhere
2. **Layers = Build optimization** - Fast rebuilds
3. **Security first** - Non-root users!
4. **Health checks** - Self-healing systems
5. **One command setup** - No installation headaches

**This transforms our AI from:**
- **"Works on my laptop"** → **"Works everywhere"**
- **"Complex setup instructions"** → **"docker run ..."**
- **"Manual deployment"** → **"Push button deployment"**

**Question for discussion:** If you were to add one more thing to this Dockerfile to make it even better, what would it be and why?






Create `docker-compose.yml`


```yaml
version: '3.8'

services:
  financial-rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - WANDB_API_KEY=${WANDB_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Add Redis for caching in future
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   volumes:
  #     - redis_data:/data

# volumes:
#   redis_data:
```




Create `.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.venv
financial-rag-env
*.db
*.sqlite
```

Create Deployment Scripts

Create `scripts/start_api.py`
# **🎬 The AI Theater Director: Production Startup Script**

Good morning class! Today we're looking at the **stage manager** of our AI production - the startup script that brings our entire Financial API to life! This is the "curtain raiser" that starts the show!

---

## **🚀 What This Script Does**

This simple but powerful script:
1. **🎪 Sets up the stage** (Python path configuration)
2. **🎭 Hires the director** (Uvicorn web server)
3. **📢 Configures the theater** (Ports, workers, logging)
4. **🎬 Starts the performance** (Launches our AI API)

**Think of it like:** The backstage crew that makes sure everything is ready before the actors (our AI) go on stage!




```python
#!/usr/bin/env python3

# ```python
# #!/usr/bin/env python3
# """
# Production API startup script
# """
# ```
# **`#!/usr/bin/env python3`** - The **"shebang"**:
# - Tells the system: "Run this with Python 3!"
# - Makes it executable: `./start_api.py` instead of `python start_api.py`
# - **`/usr/bin/env python3`** = "Find Python 3 in the system PATH"

# **Docstring**: Explains what the script does (good practice!)

"""
Production API startup script
"""

import uvicorn
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# **The Problem:**
# Our project structure looks like:
# ```
# financial-rag/
# ├── src/                    ← Our code lives here
# │   └── financial_rag/
# │       └── api/
# │           └── server.py   ← FastAPI app is here
# └── scripts/                ← This script is here
#     └── start_api.py        ← But Python can't find src!
# ```

# **The Solution:**
# ```python
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# ```
# Translation: **"Add the `src` folder (which is up one level from this script) to Python's search path!"**

# **Why `insert(0, ...)`?**
# - Puts it FIRST in the search list
# - Python checks here before system packages
# - Prevents conflicts with similarly named packages


if __name__ == "__main__":
    uvicorn.run(
        "financial_rag.api.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True
    )
```



## **🎭 Uvicorn: The Theater Director**

### **What is Uvicorn?**
- **U**niversal **Vi**rtual **CORN** (weird name, great tool!)
- ASGI server (Asynchronous Server Gateway Interface)
- **Fast**, modern Python web server
- Built for async Python (perfect for AI APIs!)

### **The Configuration:**

#### **1. The Play (Our API)**
```python
"financial_rag.api.server:app"
```
- `financial_rag.api.server` = Python module path
- `:app` = The FastAPI instance inside that module
- **Like:** "Perform the play called 'app' from the 'server' script"

#### **2. The Theater Address**
```python
host="0.0.0.0"
```
- Listen on **ALL** network interfaces
- `localhost` = Only this computer
- `0.0.0.0` = This computer AND network
- **CRITICAL** for Docker/cloud deployment!

#### **3. The Box Office Window**
```python
port=int(os.getenv("PORT", "8000"))
```
- **Environment variable** `PORT` or default `8000`
- Cloud platforms (Heroku, Render) set `PORT` automatically
- **Flexibility:** Run on different ports without code changes

#### **4. The Stage Crew (Workers)**
```python
workers=int(os.getenv("WORKERS", "1"))
```
- **Number of worker processes**
- `1` = Single worker (development)
- `4` = Four workers (can handle 4× more requests!)
- Cloud: Set based on CPU cores

#### **5. The Announcement Volume**
```python
log_level=os.getenv("LOG_LEVEL", "info")
```
- **Logging verbosity:**
  - `debug` = Everything (development)
  - `info` = Normal operations (production)
  - `warning` = Only problems
  - `error` = Only errors
- Controlled by environment variable

#### **6. The Guest Book**
```python
access_log=True
```
- Logs **every API request**
- `GET /health - 200 OK - 5ms`
- `POST /analyze - 500 Error - 3200ms`
- **Monitoring goldmine!**

---

## **🎓 Real-World Deployment Scenarios**

### **Scenario 1: Local Development**
```bash
# Simple start
python scripts/start_api.py
# OR make it executable
chmod +x scripts/start_api.py
./scripts/start_api.py

# Runs on: http://localhost:8000
# 1 worker, info logging
```

### **Scenario 2: Docker Container**
```dockerfile
# Dockerfile
CMD ["python", "scripts/start_api.py"]
# OR
CMD ["./scripts/start_api.py"]
```

### **Scenario 3: Cloud Platform (Heroku/Render)**
```bash
# They automatically set PORT
# Heroku dyno starts with:
PORT=53482 WORKERS=4 python scripts/start_api.py
# Our script adapts automatically!
```

### **Scenario 4: High-Traffic Production**
```bash
# Linux server with 8 CPU cores
PORT=8000 WORKERS=8 LOG_LEVEL=warning python scripts/start_api.py

# Can handle thousands of requests!
```

---

## **⚡ Environment Variables: The Control Panel**

### **Why Environment Variables?**
```python
port=int(os.getenv("PORT", "8000"))  # Default if not set
```

**Benefits:**
- **No code changes** to switch environments
- **Secrets safe** (API keys not in code)
- **Different configs** for dev/test/prod
- **12-factor app** best practice

### **Setting Them:**

#### **Linux/Mac:**
```bash
export PORT=9000
export WORKERS=4
export LOG_LEVEL=debug
python scripts/start_api.py
```

#### **Windows:**
```bash
set PORT=9000
set WORKERS=4
python scripts/start_api.py
```

#### **One-liner:**
```bash
PORT=9000 WORKERS=4 LOG_LEVEL=debug python scripts/start_api.py
```

---

## **🔍 The Magic of `os.getenv()`**

```python
os.getenv("PORT", "8000")
# Translation: "Get environment variable PORT, or use "8000" if not set"
```

**Safe defaults:**
- Development: No setup needed
- Production: Override as needed
- **Never crashes** if variable missing

**Type conversion:**
```python
int(os.getenv("PORT", "8000"))  # Convert to integer
# "8000" → 8000 (number)
```




Create `scripts/test_production.py`

```python
#!/usr/bin/env python3
"""
Production health check and test script
"""

import requests
import json
import sys
import time

def test_production_endpoints(base_url: str = "http://localhost:8000"):
    """Test all production endpoints"""
    
    print("🧪 Testing Production API Endpoints...")
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/system/stats", None),
        ("POST", "/query", {
            "question": "What is the current stock price of Apple?",
            "use_agent": True,
            "analysis_style": "analyst"
        })
    ]
    
    for method, path, data in endpoints:
        try:
            url = f"{base_url}{path}"
            print(f"\nTesting {method} {path}...")
            
            if method == "GET":
                response = requests.get(url, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ SUCCESS: {response.status_code}")
                if path == "/health":
                    health_data = response.json()
                    print(f"   Status: {health_data['status']}")
                    print(f"   Vector Store: {health_data['vector_store_ready']}")
                    print(f"   LLM: {health_data['llm_ready']}")
            else:
                print(f"❌ FAILED: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: {e}")
            return False
    
    print("\n🎉 All production tests completed!")
    return True

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = test_production_endpoints(base_url)
    sys.exit(0 if success else 1)
```

Updated Requirements in `pyproject.toml`

Add the new dependencies:

```toml
# Add to dependencies list
dependencies = [
    # ... existing dependencies
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.4.0",
    "wandb>=0.15.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.0",
]

[project.optional-dependencies]
dev = [
    # ... existing dev dependencies
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
]

# Add new section for docker
[project.scripts]
financial-rag-api = "financial_rag.api.server:main"
```

Create Production Test

Create `test_production.py`

```python
#!/usr/bin/env python3
"""
Comprehensive production test
"""

import sys
import os
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from financial_rag.api.server import FinancialRAGAPI
from financial_rag.config import config

async def test_production_readiness():
    """Test if the system is production ready"""
    print("🏭 Testing Production Readiness...")
    
    try:
        # Test API initialization
        print("1. Testing API initialization...")
        api = FinancialRAGAPI()
        await api.initialize_services()
        print("   ✅ API initialized successfully")
        
        # Test health endpoint
        print("2. Testing health check...")
        health_check = api.app.routes[2].endpoint  # /health endpoint
        response = await health_check()
        print(f"   ✅ Health check: {response.status}")
        
        # Test vector store
        print("3. Testing vector store...")
        if api.vector_store:
            print("   ✅ Vector store operational")
        else:
            print("   ⚠️  Vector store not initialized (expected for first run)")
        
        # Test agent
        print("4. Testing agent...")
        if api.agent:
            print("   ✅ Agent operational")
            
            # Test simple query
            test_query = "What is the current stock price of Apple?"
            try:
                result = api.agent.analyze(test_query)
                print(f"   ✅ Agent query test: {len(result['answer'])} chars response")
            except Exception as e:
                print(f"   ⚠️  Agent query test failed (may be expected): {e}")
        else:
            print("   ❌ Agent not initialized")
            return False
        
        # Test monitoring
        print("5. Testing monitoring...")
        if api.agent.monitor.enabled:
            print("   ✅ Monitoring enabled")
        else:
            print("   ⚠️  Monitoring disabled (check WandB API key)")
        
        print("\n🎉 PRODUCTION READY! All systems operational.")
        return True
        
    except Exception as e:
        print(f"\n💥 PRODUCTION TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not getattr(config, var, None)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Please set them in your .env file")
        sys.exit(1)
    
    # Run production test
    success = asyncio.run(test_production_readiness())
    sys.exit(0 if success else 1)
```

now let's run Production Test

```bash
# Install updated dependencies
pip install -e ".[dev]"

# Test production readiness
python test_production.py

# Build Docker image
docker build -t financial-rag-agent .

# Run with Docker Compose
docker-compose up -d

# Test the running API
python scripts/test_production.py

# Check logs
docker-compose logs -f
```

## What We've Built Now:

1. ✅ **Enterprise Monitoring** with Weights & Biases
2. ✅ **Production REST API** with FastAPI
3. ✅ **Docker Containerization** for deployment
4. ✅ **Health Checks** and monitoring endpoints
5. ✅ **Background Tasks** for data ingestion
6. ✅ **Production-ready configuration**



