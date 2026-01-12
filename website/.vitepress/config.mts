import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'LocalGraph',
    description: 'AI Agents 101 — Build, Learn & Experiment Locally',
    base: '/langgraph-ollama-tutorial/',

    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
      ['meta', { name: 'theme-color', content: '#10b981' }],
      ['meta', { name: 'og:type', content: 'website' }],
      ['meta', { name: 'og:title', content: 'LocalGraph — AI Agents 101' }],
      ['meta', { name: 'og:description', content: 'Build, Learn & Experiment with AI Agents Locally' }],
      ['meta', { name: 'og:image', content: '/logo.svg' }],
    ],

    themeConfig: {
      logo: '/logo.svg',

      nav: [
        { text: 'Home', link: '/' },
        { text: 'Tutorials', link: '/tutorials/' },
        { text: 'Progress', link: '/progress' },
        { text: 'Playground', link: '/playground/' },
        { text: 'API Reference', link: '/api/' },
        {
          text: 'Resources',
          items: [
            { text: 'GitHub', link: 'https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial' },
            { text: 'LangGraph Docs', link: 'https://langchain-ai.github.io/langgraph/' },
            { text: 'Ollama', link: 'https://ollama.ai' },
          ]
        }
      ],

      sidebar: {
        '/tutorials/': [
          {
            text: 'Getting Started',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/tutorials/' },
              { text: 'Setup & Installation', link: '/tutorials/setup' },
            ]
          },
          {
            text: 'Phase 1: Core Patterns',
            collapsed: false,
            items: [
              { text: '01. Chatbot Basics', link: '/tutorials/core/01-chatbot-basics' },
              { text: '02. Tool Calling (ReAct)', link: '/tutorials/core/02-tool-calling' },
              { text: '03. Memory & Persistence', link: '/tutorials/core/03-memory-persistence' },
              { text: '04. Human-in-the-Loop', link: '/tutorials/core/04-human-in-the-loop' },
              { text: '05. Reflection', link: '/tutorials/core/05-reflection' },
              { text: '06. Plan & Execute', link: '/tutorials/core/06-plan-and-execute' },
              { text: '07. Research Assistant', link: '/tutorials/core/07-research-assistant' },
            ]
          },
          {
            text: 'Phase 2: RAG Patterns',
            collapsed: false,
            items: [
              { text: '08. Basic RAG', link: '/tutorials/rag/08-basic-rag' },
              { text: '09. Self-RAG', link: '/tutorials/rag/09-self-rag' },
              { text: '10. Corrective RAG', link: '/tutorials/rag/10-crag' },
              { text: '11. Adaptive RAG', link: '/tutorials/rag/11-adaptive-rag' },
              { text: '12. Agentic RAG', link: '/tutorials/rag/12-agentic-rag' },
              { text: '13. Perplexity Clone', link: '/tutorials/rag/13-perplexity-clone' },
            ]
          },
          {
            text: 'Phase 3: Multi-Agent Patterns',
            collapsed: false,
            items: [
              { text: '14. Multi-Agent Collab', link: '/tutorials/multi-agent/14-multi-agent-collaboration' },
              { text: '15. Hierarchical Teams', link: '/tutorials/multi-agent/15-hierarchical-teams' },
              { text: '16. Subgraphs', link: '/tutorials/multi-agent/16-subgraphs' },
              { text: '17. Agent Handoffs', link: '/tutorials/multi-agent/17-agent-handoffs' },
              { text: '18. Agent Swarm', link: '/tutorials/multi-agent/18-agent-swarm' },
              { text: '19. Map-Reduce', link: '/tutorials/multi-agent/19-map-reduce-agents' },
              { text: '20. Evaluation', link: '/tutorials/multi-agent/20-multi-agent-evaluation' },
            ]
          },
          {
            text: 'Phase 4: Advanced Reasoning',
            collapsed: false,
            items: [
              { text: '21. Plan-and-Execute', link: '/tutorials/advanced/21-plan-and-execute' },
              { text: '22. Reflection', link: '/tutorials/advanced/22-reflection' },
              { text: '23. Reflexion', link: '/tutorials/advanced/23-reflexion' },
              { text: '24. LATS (Tree Search)', link: '/tutorials/advanced/24-lats' },
              { text: '25. ReWOO', link: '/tutorials/advanced/25-rewoo' },
            ]
          },
        ],
        '/playground/': [
          {
            text: 'Playground',
            items: [
              { text: 'Python Sandbox', link: '/playground/' },
              { text: 'Interactive Notebooks', link: '/playground/notebooks' },
              { text: 'Graph Visualizer', link: '/playground/visualizer' },
            ]
          }
        ],
        '/api/': [
          {
            text: 'API Reference',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/api/' },
              { text: 'Configuration', link: '/api/configuration' },
              { text: 'RAG', link: '/api/rag' },
              { text: 'Multi-Agent', link: '/api/agents' },
              { text: 'Patterns', link: '/api/patterns' },
              { text: 'State Types', link: '/api/types' },
            ]
          }
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial' }
      ],

      search: {
        provider: 'local',
        options: {
          detailedView: true
        }
      },

      footer: {
        message: 'Released under the MIT License.',
        copyright: 'Made by <a href="https://abhinaavramesh.github.io/portfolio/" target="_blank">Abhinaav Ramesh</a>'
      },

      editLink: {
        pattern: 'https://github.com/AbhinaavRamesh/langgraph-ollama-tutorial/edit/main/website/:path',
        text: 'Edit this page on GitHub'
      },

      lastUpdated: {
        text: 'Last updated',
        formatOptions: {
          dateStyle: 'medium',
          timeStyle: 'short'
        }
      }
    },

    markdown: {
      lineNumbers: true,
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      }
    },

    mermaid: {
      theme: 'default',
      darkMode: true
    },

    vite: {
      optimizeDeps: {
        include: ['mermaid']
      }
    }
  })
)
