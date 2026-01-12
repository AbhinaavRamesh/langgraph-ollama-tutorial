<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

interface AttemptRecord {
  question: string
  questionId: string
  correct: boolean
  hintsUsed: number
  textAnswer?: string
  timestamp: number
  timeTaken?: number
}

interface TutorialProgress {
  completed: boolean
  completedAt?: number
  quizPassed?: boolean
  quizScore?: {
    correct: number
    total: number
  }
}

interface WeakArea {
  questionId: string
  question: string
  wrongAttempts: number
  totalAttempts: number
  lastAttempt: number
  tutorialId: string
}

const attempts = ref<AttemptRecord[]>([])
const tutorialProgress = ref<Record<string, TutorialProgress>>({})
const currentStreak = ref(0)
const longestStreak = ref(0)

const tutorials = [
  { id: '01-chatbot-basics', phase: 'core', name: 'Chatbot Basics' },
  { id: '02-tool-calling', phase: 'core', name: 'Tool Calling' },
  { id: '03-memory-persistence', phase: 'core', name: 'Memory & Persistence' },
  { id: '04-human-in-the-loop', phase: 'core', name: 'Human in the Loop' },
  { id: '05-reflection', phase: 'core', name: 'Reflection' },
  { id: '06-plan-and-execute', phase: 'core', name: 'Plan and Execute' },
  { id: '07-research-assistant', phase: 'core', name: 'Research Assistant' },
  { id: '08-basic-rag', phase: 'rag', name: 'Basic RAG' },
  { id: '09-self-rag', phase: 'rag', name: 'Self-RAG' },
  { id: '10-crag', phase: 'rag', name: 'CRAG' },
  { id: '11-adaptive-rag', phase: 'rag', name: 'Adaptive RAG' },
  { id: '12-agentic-rag', phase: 'rag', name: 'Agentic RAG' },
  { id: '13-perplexity-clone', phase: 'rag', name: 'Perplexity Clone' },
  { id: '14-multi-agent-collaboration', phase: 'multi-agent', name: 'Multi-Agent Collaboration' },
  { id: '15-hierarchical-teams', phase: 'multi-agent', name: 'Hierarchical Teams' },
  { id: '16-subgraphs', phase: 'multi-agent', name: 'Subgraphs' },
  { id: '17-agent-handoffs', phase: 'multi-agent', name: 'Agent Handoffs' },
  { id: '18-agent-swarm', phase: 'multi-agent', name: 'Agent Swarm' },
  { id: '19-map-reduce-agents', phase: 'multi-agent', name: 'Map-Reduce Agents' },
  { id: '20-multi-agent-evaluation', phase: 'multi-agent', name: 'Multi-Agent Evaluation' },
  { id: '21-plan-and-execute', phase: 'advanced', name: 'Advanced Plan & Execute' },
  { id: '22-reflection', phase: 'advanced', name: 'Advanced Reflection' },
  { id: '23-reflexion', phase: 'advanced', name: 'Reflexion' },
  { id: '24-lats', phase: 'advanced', name: 'LATS' },
  { id: '25-rewoo', phase: 'advanced', name: 'ReWOO' }
]

const phases = [
  { id: 'core', name: 'Core Patterns', color: '#3b82f6' },
  { id: 'rag', name: 'RAG Patterns', color: '#f59e0b' },
  { id: 'multi-agent', name: 'Multi-Agent', color: '#8b5cf6' },
  { id: 'advanced', name: 'Advanced', color: '#ec4899' }
]

const totalAttempts = computed(() => attempts.value.length)
const correctAttempts = computed(() => attempts.value.filter(a => a.correct).length)
const overallAccuracy = computed(() => {
  if (totalAttempts.value === 0) return 0
  return Math.round((correctAttempts.value / totalAttempts.value) * 100)
})

const averageTimePerQuestion = computed(() => {
  const validAttempts = attempts.value.filter(a => a.timeTaken && a.timeTaken > 0)
  if (validAttempts.length === 0) return 0
  const avgMs = validAttempts.reduce((sum, a) => sum + (a.timeTaken || 0), 0) / validAttempts.length
  return Math.round(avgMs / 1000)
})

const phaseStats = computed(() => {
  return phases.map(phase => {
    const phaseTutorials = tutorials.filter(t => t.phase === phase.id)
    const phaseAttempts = attempts.value.filter(a => {
      const tutorialId = a.questionId.split('-').slice(0, 2).join('-')
      return phaseTutorials.some(t => t.id === tutorialId)
    })
    const correct = phaseAttempts.filter(a => a.correct).length
    const total = phaseAttempts.length
    const completed = phaseTutorials.filter(t => tutorialProgress.value[t.id]?.completed).length
    return {
      ...phase,
      correct,
      total,
      accuracy: total > 0 ? Math.round((correct / total) * 100) : 0,
      completed,
      totalTutorials: phaseTutorials.length,
      progress: phaseTutorials.length > 0 ? Math.round((completed / phaseTutorials.length) * 100) : 0
    }
  })
})

const weakAreas = computed(() => {
  const questionMap = new Map<string, WeakArea>()
  attempts.value.forEach(attempt => {
    const existing = questionMap.get(attempt.questionId)
    const tutorialId = attempt.questionId.split('-').slice(0, 2).join('-')
    if (existing) {
      existing.totalAttempts++
      if (!attempt.correct) existing.wrongAttempts++
      if (attempt.timestamp > existing.lastAttempt) existing.lastAttempt = attempt.timestamp
    } else {
      questionMap.set(attempt.questionId, {
        questionId: attempt.questionId,
        question: attempt.question,
        wrongAttempts: attempt.correct ? 0 : 1,
        totalAttempts: 1,
        lastAttempt: attempt.timestamp,
        tutorialId
      })
    }
  })
  return Array.from(questionMap.values())
    .filter(q => q.wrongAttempts >= 2)
    .sort((a, b) => b.wrongAttempts - a.wrongAttempts)
    .slice(0, 5)
})

onMounted(() => {
  const savedAttempts = localStorage.getItem('quizAttempts')
  if (savedAttempts) attempts.value = JSON.parse(savedAttempts)
  const savedProgress = localStorage.getItem('tutorialProgress')
  if (savedProgress) tutorialProgress.value = JSON.parse(savedProgress)
  calculateStreak()
})

const calculateStreak = () => {
  if (attempts.value.length === 0) return
  const dates = new Set(
    attempts.value.map(a => {
      const date = new Date(a.timestamp)
      return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`
    })
  )
  const sortedDates = Array.from(dates)
    .map(d => {
      const [year, month, day] = d.split('-').map(Number)
      return new Date(year, month, day)
    })
    .sort((a, b) => b.getTime() - a.getTime())
  if (sortedDates.length === 0) return
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  let streak = 0
  const mostRecent = sortedDates[0]
  mostRecent.setHours(0, 0, 0, 0)
  if (mostRecent.getTime() === today.getTime() || mostRecent.getTime() === yesterday.getTime()) {
    streak = 1
    let currentDate = new Date(mostRecent)
    for (let i = 1; i < sortedDates.length; i++) {
      const prevDate = sortedDates[i]
      prevDate.setHours(0, 0, 0, 0)
      const expectedDate = new Date(currentDate)
      expectedDate.setDate(expectedDate.getDate() - 1)
      if (prevDate.getTime() === expectedDate.getTime()) {
        streak++
        currentDate = prevDate
      } else break
    }
  }
  currentStreak.value = streak
  longestStreak.value = Math.max(streak, longestStreak.value)
}

const getTutorialName = (tutorialId: string): string => {
  const tutorial = tutorials.find(t => t.id === tutorialId)
  return tutorial ? tutorial.name : tutorialId
}

const formatTime = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}
</script>

<template>
  <div class="dashboard">
    <!-- Stats Grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Questions</div>
        <div class="stat-value">{{ totalAttempts }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Accuracy</div>
        <div class="stat-value">{{ overallAccuracy }}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Time</div>
        <div class="stat-value">{{ formatTime(averageTimePerQuestion) }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Streak</div>
        <div class="stat-value">{{ currentStreak }} <span class="stat-sub">days</span></div>
      </div>
    </div>

    <!-- Phase Progress -->
    <div class="section">
      <h3 class="section-title">Progress by Phase</h3>
      <div class="phase-list">
        <div v-for="phase in phaseStats" :key="phase.id" class="phase-row">
          <div class="phase-info">
            <span class="phase-indicator" :style="{ background: phase.color }"></span>
            <span class="phase-name">{{ phase.name }}</span>
            <span class="phase-count">{{ phase.completed }}/{{ phase.totalTutorials }}</span>
          </div>
          <div class="phase-bar-container">
            <div class="phase-bar" :style="{ width: `${phase.progress}%`, background: phase.color }"></div>
          </div>
          <div class="phase-accuracy" v-if="phase.total > 0">
            {{ phase.accuracy }}% accuracy
          </div>
          <div class="phase-accuracy" v-else>
            No attempts
          </div>
        </div>
      </div>
    </div>

    <!-- Weak Areas -->
    <div class="section" v-if="weakAreas.length > 0">
      <h3 class="section-title">Areas to Review</h3>
      <div class="weak-list">
        <div v-for="area in weakAreas" :key="area.questionId" class="weak-item">
          <div class="weak-question">{{ area.question }}</div>
          <div class="weak-meta">
            <span class="weak-tutorial">{{ getTutorialName(area.tutorialId) }}</span>
            <span class="weak-stats">{{ area.wrongAttempts }} incorrect</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="totalAttempts === 0" class="empty-state">
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
      </svg>
      <p>Complete quizzes in tutorials to track your progress</p>
    </div>
  </div>
</template>

<style scoped>
.dashboard {
  max-width: 800px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.stat-card {
  padding: 20px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}

.stat-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  line-height: 1;
}

.stat-sub {
  font-size: 0.875rem;
  font-weight: 400;
  color: var(--vp-c-text-2);
}

.section {
  margin-bottom: 32px;
}

.section-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0 0 16px 0;
}

.phase-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.phase-row {
  display: grid;
  grid-template-columns: 200px 1fr 100px;
  gap: 16px;
  align-items: center;
  padding: 12px 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}

.phase-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.phase-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.phase-name {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.phase-count {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  margin-left: auto;
}

.phase-bar-container {
  height: 6px;
  background: var(--vp-c-divider);
  border-radius: 3px;
  overflow: hidden;
}

.phase-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.phase-accuracy {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
  text-align: right;
}

.weak-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.weak-item {
  padding: 12px 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-danger-1);
  border-radius: 8px;
}

.weak-question {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
  margin-bottom: 4px;
}

.weak-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.weak-stats {
  color: var(--vp-c-danger-1);
}

.empty-state {
  text-align: center;
  padding: 48px 24px;
  color: var(--vp-c-text-2);
}

.empty-state svg {
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state p {
  margin: 0;
  font-size: 0.875rem;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .phase-row {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  .phase-accuracy {
    text-align: left;
  }
}

/* Dark mode support */
.dark .stat-card {
  background: var(--vp-c-bg-alt);
  border-color: var(--vp-c-divider);
}

.dark .phase-row {
  background: var(--vp-c-bg-alt);
}

.dark .weak-item {
  background: var(--vp-c-bg-alt);
}

.dark .phase-bar-container {
  background: var(--vp-c-bg-alt);
}
</style>
