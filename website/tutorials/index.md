---
title: LocalGraph Tutorials - AI Agents 101
description: Master AI agent development with 25 hands-on tutorials. Build locally with LangGraph and Ollama — no cloud APIs required.
---

<script setup>
import TutorialCard from '../.vitepress/theme/components/TutorialCard.vue'
import PhaseProgress from '../.vitepress/theme/components/PhaseProgress.vue'
</script>

# LocalGraph Tutorials

Welcome to AI Agents 101! Master production-ready AI agents that run entirely on your machine using LangGraph and Ollama.

## What You'll Build

Through 25 hands-on tutorials, you'll master the patterns and techniques used in production AI agents:

- **Core Patterns**: Build conversational agents with tools, memory, and human oversight
- **RAG Systems**: Create search and research assistants with document retrieval
- **Multi-Agent Systems**: Coordinate specialized agents for complex workflows
- **Advanced Reasoning**: Implement cutting-edge patterns like Reflexion and LATS

Each tutorial includes:
- Step-by-step implementation guide
- Working code examples
- Interactive playground to test your agent
- Quiz to verify understanding
- Progress tracking across your learning journey

## Getting Started

New to LangGraph? Start here:

1. [Setup Guide](./setup.md) - Install dependencies and verify your environment
2. [Tutorial 01: Chatbot Basics](./core/01-chatbot-basics.md) - Learn LangGraph fundamentals
3. Work through tutorials sequentially or jump to patterns that interest you

---

## Phase 1: Core Patterns

<PhaseProgress
  phase="core"
  :tutorials="[
    '01-chatbot-basics',
    '02-tool-calling',
    '03-memory-persistence',
    '04-human-in-the-loop',
    '05-reflection',
    '06-plan-and-execute',
    '07-research-assistant'
  ]"
/>

Master the fundamental patterns that power all LangGraph applications. These tutorials teach you how to build conversational agents with tool calling, memory, human oversight, and self-improvement capabilities.

<div class="tutorial-grid">

<TutorialCard
  :number="1"
  title="Chatbot Basics"
  description="Learn StateGraph, nodes, edges, and message handling. Build your first conversational agent with streaming support."
  link="./core/01-chatbot-basics"
  phase="core"
/>

<TutorialCard
  :number="2"
  title="Tool Calling"
  description="Implement the ReAct pattern from scratch. Give your agent access to tools and let it decide when to use them."
  link="./core/02-tool-calling"
  phase="core"
/>

<TutorialCard
  :number="3"
  title="Memory & Persistence"
  description="Add conversation memory with checkpointers. Implement thread-based conversations that persist across sessions."
  link="./core/03-memory-persistence"
  phase="core"
/>

<TutorialCard
  :number="4"
  title="Human-in-the-Loop"
  description="Pause execution for human approval. Implement approval workflows and resume from saved checkpoints."
  link="./core/04-human-in-the-loop"
  phase="core"
/>

<TutorialCard
  :number="5"
  title="Reflection"
  description="Build generate-critique-revise loops. Teach your agent to improve its own outputs through self-criticism."
  link="./core/05-reflection"
  phase="core"
/>

<TutorialCard
  :number="6"
  title="Plan & Execute"
  description="Separate planning from execution. Use structured outputs to build multi-step workflows with replanning."
  link="./core/06-plan-and-execute"
  phase="core"
/>

<TutorialCard
  :number="7"
  title="Research Assistant"
  description="Capstone project combining all core patterns. Build a research assistant with web search and citation tracking."
  link="./core/07-research-assistant"
  phase="core"
/>

</div>

---

## Phase 2: RAG Patterns

<PhaseProgress
  phase="rag"
  :tutorials="[
    '08-basic-rag',
    '09-self-rag',
    '10-crag',
    '11-adaptive-rag',
    '12-agentic-rag',
    '13-perplexity-clone'
  ]"
/>

Learn to build retrieval-augmented generation systems that ground LLM responses in your documents. Progress from basic retrieval to advanced patterns with quality grading, web search fallbacks, and agentic control.

<div class="tutorial-grid">

<TutorialCard
  :number="8"
  title="Basic RAG"
  description="Document loading, chunking, embeddings, and ChromaDB. Build a simple question-answering system."
  link="./rag/08-basic-rag"
  phase="rag"
/>

<TutorialCard
  :number="9"
  title="Self-RAG"
  description="Add document grading and hallucination detection. Implement retry loops for better answer quality."
  link="./rag/09-self-rag"
  phase="rag"
/>

<TutorialCard
  :number="10"
  title="CRAG (Corrective RAG)"
  description="Web search fallback when documents aren't relevant. Combine local retrieval with external knowledge."
  link="./rag/10-crag"
  phase="rag"
/>

<TutorialCard
  :number="11"
  title="Adaptive RAG"
  description="Query classification and strategy routing. Choose between retrieval, web search, or direct generation."
  link="./rag/11-adaptive-rag"
  phase="rag"
/>

<TutorialCard
  :number="12"
  title="Agentic RAG"
  description="Agent-controlled retrieval with multi-step reasoning. Let the agent decide when and what to retrieve."
  link="./rag/12-agentic-rag"
  phase="rag"
/>

<TutorialCard
  :number="13"
  title="Perplexity Clone"
  description="Build a research assistant with citations and source metadata. Implement follow-up question flows."
  link="./rag/13-perplexity-clone"
  phase="rag"
/>

</div>

---

## Phase 3: Multi-Agent Patterns

<PhaseProgress
  phase="multi-agent"
  :tutorials="[
    '14-multi-agent-collaboration',
    '15-hierarchical-teams',
    '16-subgraphs',
    '17-agent-handoffs',
    '18-agent-swarm',
    '19-map-reduce-agents',
    '20-multi-agent-evaluation'
  ]"
/>

Coordinate multiple specialized agents to tackle complex tasks. Learn patterns for collaboration, hierarchies, handoffs, and evaluation of multi-agent systems.

<div class="tutorial-grid">

<TutorialCard
  :number="14"
  title="Multi-Agent Collaboration"
  description="Supervisor coordinates researcher, coder, and reviewer agents. Learn agent orchestration patterns."
  link="./multi-agent/14-multi-agent-collaboration"
  phase="multi-agent"
/>

<TutorialCard
  :number="15"
  title="Hierarchical Teams"
  description="Nested team structures with managers and workers. Build scalable multi-agent hierarchies."
  link="./multi-agent/15-hierarchical-teams"
  phase="multi-agent"
/>

<TutorialCard
  :number="16"
  title="Subgraphs"
  description="Composable graph components for reusable agent teams. Encapsulate complex workflows."
  link="./multi-agent/16-subgraphs"
  phase="multi-agent"
/>

<TutorialCard
  :number="17"
  title="Agent Handoffs"
  description="Seamless transitions between specialized agents. Implement context-preserving handoff patterns."
  link="./multi-agent/17-agent-handoffs"
  phase="multi-agent"
/>

<TutorialCard
  :number="18"
  title="Agent Swarm"
  description="Decentralized agent coordination without supervisors. Emergent behavior from agent interactions."
  link="./multi-agent/18-agent-swarm"
  phase="multi-agent"
/>

<TutorialCard
  :number="19"
  title="Map-Reduce Agents"
  description="Parallel processing with result aggregation. Scale work across multiple agents efficiently."
  link="./multi-agent/19-map-reduce-agents"
  phase="multi-agent"
/>

<TutorialCard
  :number="20"
  title="Multi-Agent Evaluation"
  description="Testing and debugging multi-agent systems. Metrics, logging, and quality assurance strategies."
  link="./multi-agent/20-multi-agent-evaluation"
  phase="multi-agent"
/>

</div>

---

## Phase 4: Advanced Reasoning

<PhaseProgress
  phase="advanced"
  :tutorials="[
    '21-plan-and-execute',
    '22-reflection',
    '23-reflexion',
    '24-lats',
    '25-rewoo'
  ]"
/>

Implement cutting-edge research patterns for complex reasoning tasks. These advanced techniques push the boundaries of what local LLMs can accomplish.

<div class="tutorial-grid">

<TutorialCard
  :number="21"
  title="Plan-and-Execute"
  description="Strategic planning before tactical execution. Upfront planning with optional replanning based on results."
  link="./advanced/21-plan-and-execute"
  phase="advanced"
/>

<TutorialCard
  :number="22"
  title="Reflection"
  description="Advanced self-improvement patterns. Multiple critique cycles for high-quality outputs."
  link="./advanced/22-reflection"
  phase="advanced"
/>

<TutorialCard
  :number="23"
  title="Reflexion"
  description="Learning from trial and error. Verbal reinforcement learning with episodic memory."
  link="./advanced/23-reflexion"
  phase="advanced"
/>

<TutorialCard
  :number="24"
  title="LATS (Tree Search)"
  description="Language agent tree search with value functions. Explore multiple reasoning paths."
  link="./advanced/24-lats"
  phase="advanced"
/>

<TutorialCard
  :number="25"
  title="ReWOO"
  description="Reasoning without observation overhead. Efficient tool use with decoupled planning."
  link="./advanced/25-rewoo"
  phase="advanced"
/>

</div>

---

## Learning Path Recommendations

### For Beginners
Start with Core Patterns (1-7) and complete them sequentially. These build on each other and establish the foundation for all other patterns.

### For RAG Applications
Complete Core Patterns 1-3, then move to RAG Patterns (8-13). Focus on understanding retrieval quality and when to use each pattern.

### For Multi-Agent Systems
Complete Core Patterns 1-7 first, then explore Multi-Agent Patterns (14-20). Understanding single-agent patterns is essential before coordination.

### For Research Implementation
Complete all Core Patterns, then jump to Advanced Reasoning (21-25). These implement cutting-edge research papers and require solid fundamentals.

---

## Additional Resources

- [Setup Guide](./setup.md) - Installation and configuration
- [GitHub Repository](https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial) - Source code and examples
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) - Official LangGraph docs
- [Ollama Models](https://ollama.com/library) - Browse available models

## Need Help?

- Check the [Setup Guide](./setup.md) for troubleshooting
- Review tutorial prerequisites before starting
- Test your environment with `langgraph-local check`
- Each tutorial includes common pitfalls and solutions

Ready to start? Head to the [Setup Guide](./setup.md) to prepare your environment.

<style scoped>
.tutorial-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

@media (max-width: 768px) {
  .tutorial-grid {
    grid-template-columns: 1fr;
  }
}
</style>
