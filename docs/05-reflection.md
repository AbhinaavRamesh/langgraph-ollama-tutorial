# Tutorial 05: Reflection

This tutorial teaches how to build self-critiquing agents that iteratively improve their outputs through reflection loops.

## What You'll Learn

- **Reflection loops**: Generate → Critique → Revise patterns
- **Self-improvement**: Using LLMs to evaluate their own outputs
- **Iteration control**: When to stop refining
- **Quality enhancement**: Producing better outputs through feedback

## Prerequisites

- Completed [Tutorial 04: Human-in-the-Loop](04-human-in-the-loop.md)
- Understanding of conditional edges

---

## What is Reflection?

Reflection is a pattern where an LLM:
1. **Generates** an initial output
2. **Critiques** its own work
3. **Revises** based on the critique
4. **Repeats** until satisfied

This mirrors how humans improve their work through drafts and revisions.

### Why Use Reflection?

Single-shot LLM outputs are often good but not great. Through reflection:
- Errors get caught and corrected
- Missing information gets added
- Clarity improves with each revision
- Quality approaches human-level editing

### Use Cases

- **Writing**: Essays, reports, documentation
- **Code generation**: Write, review, refactor
- **Analysis**: Initial assessment → deeper analysis → conclusions
- **Problem-solving**: Solution → evaluation → refinement

---

## Core Concepts

### 1. The Reflection Loop

![Reflection Graph](images/05-reflection-graph.png)

The graph cycles between generation and critique until approved or max iterations reached.

### 2. State for Reflection

We track more than just messages:

```python
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]  # History
    task: str           # Original task
    draft: str          # Current draft
    critique: str       # Latest critique
    iteration: int      # Current iteration
```

### 3. Stopping Conditions

Two common ways to exit the loop:
1. **Approval signal**: Critique says "APPROVED" or "No changes needed"
2. **Max iterations**: Prevent infinite loops (e.g., max 3 iterations)

```python
def should_continue(state):
    if "APPROVED" in state["critique"].upper():
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "generate"
```

---

## Building a Reflection Agent

### Step 1: Define State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int
```

### Step 2: Create Generator Node

```python
from langchain_core.messages import HumanMessage, SystemMessage

GENERATOR_PROMPT = """You are a skilled writer.
If this is the first draft, write a complete response.
If you have critique, revise your draft to address the feedback."""

def generate_node(state: ReflectionState) -> dict:
    iteration = state.get("iteration", 0)

    if iteration == 0:
        prompt = f"Write a response: {state['task']}"
    else:
        prompt = f"Revise based on critique:\nDraft: {state['draft']}\nCritique: {state['critique']}"

    response = llm.invoke([
        SystemMessage(content=GENERATOR_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {
        "draft": response.content,
        "iteration": iteration + 1
    }
```

### Step 3: Create Critique Node

```python
CRITIQUE_PROMPT = """You are a thoughtful editor.
If the draft is excellent, respond with exactly: "APPROVED"
Otherwise, provide specific feedback for improvement."""

def critique_node(state: ReflectionState) -> dict:
    prompt = f"Critique this draft:\n{state['draft']}"

    response = llm.invoke([
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=prompt)
    ])

    return {"critique": response.content}
```

### Step 4: Define Routing

```python
MAX_ITERATIONS = 3

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "generate"
```

### Step 5: Build Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ReflectionState)

workflow.add_node("generate", generate_node)
workflow.add_node("critique", critique_node)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "critique")
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {"generate": "generate", "end": END}
)

graph = workflow.compile()
```

### Step 6: Use It

```python
result = graph.invoke({
    "task": "Explain LangGraph in 2 sentences.",
    "messages": [],
    "draft": "",
    "critique": "",
    "iteration": 0
})

print(result["draft"])  # Final, refined output
```

---

## Graph Visualization

![Reflection Graph](images/05-reflection-graph.png)

The graph cycles between generation and critique until approved or max iterations.

---

## Complete Code

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_ollama_local import LocalAgentConfig

# === State ===
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int

# === LLM ===
config = LocalAgentConfig()
llm = ChatOllama(
    model=config.ollama.model,
    base_url=config.ollama.base_url,
    temperature=0.7,
)

# === Nodes ===
def generate(state: ReflectionState) -> dict:
    iteration = state.get("iteration", 0)
    if iteration == 0:
        prompt = f"Write a response: {state['task']}"
    else:
        prompt = f"Revise based on critique:\nDraft: {state['draft']}\nCritique: {state['critique']}"

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "iteration": iteration + 1}

def critique(state: ReflectionState) -> dict:
    prompt = f"Critique this (say APPROVED if perfect):\n{state['draft']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": response.content}

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    if state["iteration"] >= 3:
        return "end"
    return "generate"

# === Graph ===
workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate)
workflow.add_node("critique", critique)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "critique")
workflow.add_conditional_edges("critique", should_continue, {"generate": "generate", "end": END})
graph = workflow.compile()

# === Use ===
result = graph.invoke({
    "task": "Explain recursion in 2 sentences.",
    "messages": [],
    "draft": "",
    "critique": "",
    "iteration": 0
})
print(result["draft"])
```

---

## Variations

### Two-Model Reflection

Use a stronger model for critique:

```python
generator = ChatOllama(model="llama3.2:3b")  # Fast
critic = ChatOllama(model="llama3.1:70b")    # Thorough
```

### Structured Feedback

Use JSON for specific improvement areas:

```python
CRITIQUE_PROMPT = """Return JSON with:
{
    "approved": true/false,
    "clarity": "feedback on clarity",
    "accuracy": "feedback on accuracy",
    "completeness": "feedback on completeness"
}"""
```

### Tool-Augmented Reflection

Add research tools to fact-check claims before approving.

---

## Common Pitfalls

### 1. Infinite Loops

```python
# WRONG - no termination condition
def should_continue(state):
    return "generate"  # Always continues!

# CORRECT - multiple exit conditions
MAX_ITERATIONS = 3

def should_continue(state):
    # Exit on approval
    if "APPROVED" in state.get("critique", "").upper():
        return "end"
    # Exit on max iterations
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    # Continue refining
    return "generate"
```

### 2. Critique Ignoring Instructions

```python
# WRONG - vague critique prompt
"Give feedback on this draft"

# CORRECT - explicit approval signal
CRITIQUE_PROMPT = """You are a thoughtful editor.

Evaluate this draft against these criteria:
1. Clarity - Is the message clear?
2. Accuracy - Are all claims correct?
3. Completeness - Is anything missing?

If the draft meets all criteria, respond with exactly: "APPROVED"
Otherwise, provide specific, actionable feedback for each issue."""
```

### 3. Generator Not Using Critique

```python
# WRONG - ignoring previous critique
def generate(state):
    prompt = f"Write about: {state['task']}"  # No reference to critique
    ...

# CORRECT - incorporate critique
def generate(state):
    if state["iteration"] == 0:
        prompt = f"Write about: {state['task']}"
    else:
        prompt = f"""Revise this draft to address the critique:

Original task: {state['task']}
Current draft: {state['draft']}
Critique to address: {state['critique']}

Produce an improved version that specifically addresses each point in the critique."""
    ...
```

### 4. Not Tracking Iterations

```python
# WRONG - no iteration count
class State(TypedDict):
    draft: str
    critique: str
    # Missing iteration!

# CORRECT - track iteration count
class State(TypedDict):
    draft: str
    critique: str
    iteration: int  # Essential for termination

def generate(state):
    return {
        "draft": new_draft,
        "iteration": state.get("iteration", 0) + 1
    }
```

### 5. Overly Strict Critique

```python
# WRONG - critique never approves
# "This could always be improved..."

# CORRECT - define clear approval threshold
CRITIQUE_PROMPT = """Evaluate the draft.

Approval threshold:
- Minor stylistic preferences are NOT grounds for rejection
- Focus on factual accuracy and completeness
- If the draft is "good enough" for the intended purpose, approve it

If approved: respond with "APPROVED"
If issues found: list ONLY critical issues"""
```

---

## Testing Reflection

### Unit Testing the Generate Node

```python
def test_generate_first_iteration():
    """Test initial generation."""
    state = {
        "task": "Explain recursion in 2 sentences",
        "draft": "",
        "critique": "",
        "iteration": 0
    }

    result = generate(state)

    assert "draft" in result
    assert len(result["draft"]) > 0
    assert result["iteration"] == 1

def test_generate_revision():
    """Test revision based on critique."""
    state = {
        "task": "Explain recursion",
        "draft": "Recursion is a function that calls itself.",
        "critique": "Add an example and explain the base case.",
        "iteration": 1
    }

    result = generate(state)

    # Should incorporate feedback
    assert "draft" in result
    # Draft should be different/longer after revision
    assert len(result["draft"]) > len(state["draft"])
```

### Testing the Critique Node

```python
def test_critique_approves_good_draft():
    """Test that good drafts get approved."""
    state = {
        "draft": "Recursion is when a function calls itself to solve a problem. "
                 "For example, calculating factorial: factorial(n) = n * factorial(n-1), "
                 "with base case factorial(0) = 1."
    }

    result = critique(state)

    # Should approve well-structured explanation
    assert "APPROVED" in result["critique"].upper() or \
           "complete" in result["critique"].lower()

def test_critique_identifies_issues():
    """Test that incomplete drafts get feedback."""
    state = {
        "draft": "Recursion is when a function calls itself."
    }

    result = critique(state)

    # Should identify missing elements
    assert "critique" in result
    assert len(result["critique"]) > 10  # Non-trivial feedback
```

### Testing the Routing Logic

```python
def test_should_continue_approved():
    """Test termination on approval."""
    state = {"critique": "APPROVED - this is excellent", "iteration": 1}
    assert should_continue(state) == "end"

def test_should_continue_max_iterations():
    """Test termination on max iterations."""
    state = {"critique": "Needs more work", "iteration": 3}
    assert should_continue(state) == "end"

def test_should_continue_needs_revision():
    """Test continuation for revisions."""
    state = {"critique": "Add more examples", "iteration": 1}
    assert should_continue(state) == "generate"
```

### Integration Testing

```python
@pytest.mark.integration
def test_reflection_improves_quality():
    """Test that reflection produces improved output."""
    result = graph.invoke({
        "task": "Explain the Pythagorean theorem",
        "messages": [],
        "draft": "",
        "critique": "",
        "iteration": 0
    })

    # Should have gone through at least one iteration
    assert result["iteration"] >= 1

    # Final draft should be substantial
    assert len(result["draft"]) > 50

    # Should contain key concepts
    draft_lower = result["draft"].lower()
    assert "triangle" in draft_lower or "sides" in draft_lower
```

---

## Advanced Reflection Patterns

### 1. Multi-Criteria Evaluation

Evaluate against multiple dimensions:

```python
class MultiCriteriaState(TypedDict):
    task: str
    draft: str
    clarity_score: int
    accuracy_score: int
    completeness_score: int
    overall_feedback: str
    iteration: int

def multi_criteria_critique(state):
    """Evaluate draft on multiple criteria."""
    prompt = f"""Rate this draft on each criterion (1-10):

Draft: {state['draft']}

Return JSON:
{{
    "clarity": {{"score": X, "feedback": "..."}},
    "accuracy": {{"score": X, "feedback": "..."}},
    "completeness": {{"score": X, "feedback": "..."}}
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        scores = json.loads(response.content)
        return {
            "clarity_score": scores["clarity"]["score"],
            "accuracy_score": scores["accuracy"]["score"],
            "completeness_score": scores["completeness"]["score"],
            "overall_feedback": response.content
        }
    except:
        return {"overall_feedback": response.content}

def should_continue_multi_criteria(state):
    """Continue if any criterion below threshold."""
    THRESHOLD = 7

    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    if all([
        state.get("clarity_score", 0) >= THRESHOLD,
        state.get("accuracy_score", 0) >= THRESHOLD,
        state.get("completeness_score", 0) >= THRESHOLD
    ]):
        return "end"

    return "generate"
```

### 2. External Feedback Integration

Incorporate human feedback into the loop:

```python
from langgraph.types import interrupt

def critique_with_human_review(state):
    """Get LLM critique, then optionally human feedback."""
    # LLM critique
    llm_critique = llm.invoke([
        HumanMessage(content=f"Critique: {state['draft']}")
    ]).content

    # Optionally interrupt for human review
    if state["iteration"] > 0:
        human_feedback = interrupt({
            "draft": state["draft"],
            "llm_critique": llm_critique,
            "message": "Review draft and critique. Add comments or approve."
        })

        if human_feedback.get("approved"):
            return {"critique": "APPROVED by human reviewer"}

        combined = f"LLM: {llm_critique}\n\nHuman: {human_feedback.get('comments', '')}"
        return {"critique": combined}

    return {"critique": llm_critique}
```

### 3. Progressive Quality Thresholds

Increase quality bar with each iteration:

```python
def should_continue_progressive(state):
    """Higher quality bar with each iteration."""
    iteration = state["iteration"]
    score = state.get("score", 0)

    # Progressive thresholds
    thresholds = {1: 5, 2: 7, 3: 8}
    threshold = thresholds.get(iteration, 8)

    if score >= threshold:
        return "end"
    if iteration >= MAX_ITERATIONS:
        return "end"

    return "generate"
```

---

## Performance Considerations

### 1. Limit Iterations Appropriately

More iterations = higher latency and cost:

| Max Iterations | Typical Latency | Quality Improvement |
|----------------|-----------------|---------------------|
| 1 | Fast | None (single shot) |
| 2 | Medium | Significant |
| 3 | Slower | Moderate |
| 4+ | Slow | Diminishing returns |

### 2. Early Exit Conditions

Exit as soon as quality is sufficient:

```python
def should_continue(state):
    critique = state.get("critique", "")

    # Fast exit on explicit approval
    if "APPROVED" in critique.upper():
        return "end"

    # Fast exit on high score
    if state.get("score", 0) >= 9:
        return "end"

    # Exit on max iterations
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    return "generate"
```

### 3. Async Reflection for Throughput

```python
async def async_reflection(tasks: list[str]) -> list[dict]:
    """Process multiple reflection tasks concurrently."""
    async def reflect_one(task: str) -> dict:
        return await graph.ainvoke({
            "task": task,
            "draft": "",
            "critique": "",
            "iteration": 0
        })

    results = await asyncio.gather(*[reflect_one(t) for t in tasks])
    return list(results)
```

---

## Production Checklist

- [ ] **Iteration limits**: MAX_ITERATIONS set appropriately
- [ ] **Approval signals**: Clear termination conditions
- [ ] **Critique prompts**: Specific, actionable, with examples
- [ ] **Error handling**: Handle LLM failures gracefully
- [ ] **Metrics tracking**: Monitor iterations and quality scores
- [ ] **Testing**: Both approval and revision paths tested
- [ ] **Logging**: Track drafts, critiques, and iteration counts
- [ ] **Timeouts**: Set LLM call timeouts
- [ ] **Cost awareness**: Track token usage per reflection
- [ ] **Fallback**: Return best draft if max iterations reached

---

## Running the Notebook

```bash
cd examples
jupyter lab 05_reflection.ipynb
```

---

## Key Takeaways

| Concept | What It Does |
|---------|--------------|
| **Generate Node** | Creates or revises content based on task/critique |
| **Critique Node** | Evaluates quality and provides feedback |
| **Conditional Loop** | Routes back to generate or ends |
| **Iteration Limit** | Prevents infinite loops (essential!) |
| **Approval Signal** | Explicit signal to stop refining |
| **Multi-criteria** | Evaluate on multiple dimensions |

---

## What's Next?

[Tutorial 06: Plan and Execute](06-plan-and-execute.md) - Learn how to break complex tasks into steps, plan before executing, and re-plan based on results.
