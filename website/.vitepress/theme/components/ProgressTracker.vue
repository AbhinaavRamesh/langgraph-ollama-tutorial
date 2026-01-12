<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

interface QuizScore {
  correct: number
  total: number
}

interface TutorialProgress {
  completed: boolean
  completedAt?: number
  quizPassed?: boolean
  quizScore?: QuizScore
}

const progress = ref<Record<string, TutorialProgress>>({})

const tutorials = [
  // Phase 1: Core
  '01-chatbot-basics', '02-tool-calling', '03-memory-persistence',
  '04-human-in-the-loop', '05-reflection', '06-plan-and-execute', '07-research-assistant',
  // Phase 2: RAG
  '08-basic-rag', '09-self-rag', '10-crag', '11-adaptive-rag', '12-agentic-rag', '13-perplexity-clone',
  // Phase 3: Multi-Agent
  '14-multi-agent-collaboration', '15-hierarchical-teams', '16-subgraphs',
  '17-agent-handoffs', '18-agent-swarm', '19-map-reduce-agents', '20-multi-agent-evaluation',
  // Phase 4: Advanced
  '21-plan-and-execute', '22-reflection', '23-reflexion', '24-lats', '25-rewoo'
]

const phases = [
  { name: 'Core', tutorials: tutorials.slice(0, 7), color: '#1565c0' },
  { name: 'RAG', tutorials: tutorials.slice(7, 13), color: '#ef6c00' },
  { name: 'Multi-Agent', tutorials: tutorials.slice(13, 20), color: '#7b1fa2' },
  { name: 'Advanced', tutorials: tutorials.slice(20, 25), color: '#c2185b' }
]

const totalCompleted = computed(() =>
  Object.values(progress.value).filter(p => p.completed).length
)

const totalTutorials = computed(() => tutorials.length)

const percentage = computed(() =>
  Math.round((totalCompleted.value / totalTutorials.value) * 100)
)

const phaseProgress = computed(() =>
  phases.map(phase => ({
    ...phase,
    completed: phase.tutorials.filter(t => progress.value[t]?.completed).length,
    total: phase.tutorials.length
  }))
)

const currentTutorialId = computed(() => {
  const path = route.path
  const match = path.match(/(\d{2}-[\w-]+)/)
  return match ? match[1] : null
})

const isCurrentCompleted = computed(() =>
  currentTutorialId.value ? progress.value[currentTutorialId.value]?.completed : false
)

const currentQuizScore = computed(() => {
  if (!currentTutorialId.value) return null
  return progress.value[currentTutorialId.value]?.quizScore || null
})

const isQuizPassed = computed(() => {
  if (!currentTutorialId.value) return false
  return progress.value[currentTutorialId.value]?.quizPassed || false
})

const canMarkComplete = computed(() => {
  if (!currentTutorialId.value) return false
  // Can mark complete only if quiz is passed
  return isQuizPassed.value && !isCurrentCompleted.value
})

onMounted(() => {
  loadProgress()
})

onUnmounted(() => {
  window.removeEventListener('quiz-completed', handleQuizComplete as EventListener)
})

const loadProgress = () => {
  const saved = localStorage.getItem('tutorialProgress')
  if (saved) {
    progress.value = JSON.parse(saved)
  }

  // Listen for quiz completion events
  window.addEventListener('quiz-completed', handleQuizComplete as EventListener)
}

const handleQuizComplete = (event: CustomEvent) => {
  const { tutorialId, correct, total } = event.detail

  if (!tutorialId) return

  // Update quiz score
  if (!progress.value[tutorialId]) {
    progress.value[tutorialId] = {
      completed: false,
      quizPassed: false,
      quizScore: { correct: 0, total: 0 }
    }
  }

  // Update or initialize quiz score
  const currentScore = progress.value[tutorialId].quizScore || { correct: 0, total: 0 }
  progress.value[tutorialId].quizScore = {
    correct: currentScore.correct + (correct ? 1 : 0),
    total: currentScore.total + 1
  }

  // Mark quiz as passed if at least 1 correct answer
  progress.value[tutorialId].quizPassed =
    progress.value[tutorialId].quizScore!.correct >= 1

  saveProgress()
}

const saveProgress = () => {
  localStorage.setItem('tutorialProgress', JSON.stringify(progress.value))
}

const markComplete = () => {
  if (!currentTutorialId.value || !canMarkComplete.value) return

  progress.value[currentTutorialId.value] = {
    ...progress.value[currentTutorialId.value],
    completed: true,
    completedAt: Date.now()
  }
  saveProgress()
}

const markIncomplete = () => {
  if (!currentTutorialId.value) return

  delete progress.value[currentTutorialId.value]
  saveProgress()
}

// Expose for parent components
defineExpose({
  markComplete,
  markIncomplete,
  progress
})
</script>

<template>
  <div class="progress-tracker">
    <div class="progress-header">
      <h4>Your Progress</h4>
      <span class="progress-badge">{{ totalCompleted }}/{{ totalTutorials }}</span>
    </div>

    <div class="overall-progress">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${percentage}%` }"></div>
      </div>
      <span class="progress-percentage">{{ percentage }}% complete</span>
    </div>

    <div class="phase-list">
      <div
        v-for="phase in phaseProgress"
        :key="phase.name"
        class="phase-item"
      >
        <div class="phase-info">
          <span class="phase-dot" :style="{ background: phase.color }"></span>
          <span class="phase-name">{{ phase.name }}</span>
        </div>
        <span class="phase-count">{{ phase.completed }}/{{ phase.total }}</span>
      </div>
    </div>

    <div v-if="currentTutorialId" class="current-tutorial">
      <!-- Quiz Status Indicator -->
      <div v-if="currentQuizScore" class="quiz-status">
        <div class="quiz-status-header">
          <span class="quiz-status-label">Quiz Progress</span>
          <span
            v-if="isQuizPassed"
            class="quiz-status-icon passed"
            title="Quiz Passed!"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </span>
          <span
            v-else
            class="quiz-status-icon warning"
            title="Quiz not passed yet"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </span>
        </div>
        <div class="quiz-score">
          {{ currentQuizScore.correct }}/{{ currentQuizScore.total }} correct
        </div>
      </div>

      <!-- Mark Complete Button -->
      <button
        v-if="!isCurrentCompleted"
        class="complete-btn"
        :class="{ disabled: !canMarkComplete }"
        :disabled="!canMarkComplete"
        @click="markComplete"
        :title="!isQuizPassed ? 'Pass the quiz first to mark complete' : 'Mark this tutorial as complete'"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        Mark Complete
      </button>
      <button
        v-else
        class="completed-btn"
        @click="markIncomplete"
      >
        <span class="completion-check">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        </span>
        Completed!
      </button>
    </div>
  </div>
</template>

<style scoped>
.progress-tracker {
  width: 240px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.progress-header h4 {
  margin: 0;
  font-size: 0.9rem;
}

.progress-badge {
  background: var(--vp-c-brand-1);
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
}

.overall-progress {
  margin-bottom: 16px;
}

.progress-bar {
  height: 6px;
  background: var(--vp-c-divider);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-percentage {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.phase-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.phase-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
}

.phase-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.phase-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.phase-name {
  color: var(--vp-c-text-2);
}

.phase-count {
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.current-tutorial {
  padding-top: 12px;
  border-top: 1px solid var(--vp-c-divider);
}

.complete-btn, .completed-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.complete-btn {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}

.complete-btn:not(:disabled):hover {
  border-color: var(--vp-c-brand-1);
  background: rgba(62, 175, 124, 0.1);
}

.complete-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.complete-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.completed-btn {
  background: rgba(76, 175, 80, 0.1);
  border: 1px solid #4caf50;
  color: #4caf50;
}

.completed-btn:hover {
  background: rgba(76, 175, 80, 0.2);
}

/* Quiz Status Styles */
.quiz-status {
  margin-bottom: 12px;
  padding: 10px 12px;
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
}

.quiz-status-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.quiz-status-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.quiz-status-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

.quiz-status-icon.passed {
  background: rgba(76, 175, 80, 0.1);
  color: #4caf50;
}

.quiz-status-icon.warning {
  background: rgba(255, 152, 0, 0.1);
  color: #ff9800;
}

.quiz-score {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
</style>
