<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

interface QuizOption {
  text: string
  correct?: boolean
}

interface Hint {
  text: string
  penalty?: number
}

interface AttemptRecord {
  question: string
  questionId: string
  correct: boolean
  hintsUsed: number
  textAnswer?: string
  timestamp: number
  timeTaken?: number
}

const props = withDefaults(defineProps<{
  question: string
  options: QuizOption[]
  explanation?: string
  tutorialId?: string
  type?: 'multiple-choice' | 'true-false' | 'fill-blank'
  hints?: Hint[]
  acceptedAnswers?: string[]
  caseSensitive?: boolean
}>(), {
  type: 'multiple-choice',
  caseSensitive: false
})

const emit = defineEmits<{
  (e: 'complete', correct: boolean): void
  (e: 'attempt', data: { correct: boolean, hintsUsed: number }): void
  (e: 'hint-used', hintIndex: number): void
}>()

const selectedIndex = ref<number | null>(null)
const textAnswer = ref('')
const submitted = ref(false)
const hintsRevealed = ref<number>(0)
const startTime = ref<number>(Date.now())
const showHistory = ref(false)

const questionId = computed(() => {
  return `${props.tutorialId || 'quiz'}-${props.question.substring(0, 50).replace(/\s+/g, '-')}`
})

const correctIndex = computed(() =>
  props.options.findIndex(opt => opt.correct)
)

const isCorrect = computed(() => {
  if (props.type === 'fill-blank') {
    return checkFillBlankAnswer()
  }
  return selectedIndex.value === correctIndex.value
})

const isTrueFalse = computed(() => props.type === 'true-false')
const isFillBlank = computed(() => props.type === 'fill-blank')

const canSubmit = computed(() => {
  if (isFillBlank.value) return textAnswer.value.trim().length > 0
  return selectedIndex.value !== null
})

const attemptHistory = computed(() => {
  if (typeof window === 'undefined') return []
  const attempts = JSON.parse(localStorage.getItem('quizAttempts') || '[]') as AttemptRecord[]
  return attempts.filter(a => a.questionId === questionId.value).slice(-5)
})

const stats = computed(() => {
  const history = attemptHistory.value
  if (history.length === 0) return null

  const correctCount = history.filter(a => a.correct).length
  const totalAttempts = history.length
  const correctRate = (correctCount / totalAttempts * 100).toFixed(0)

  const avgTime = history.reduce((sum, a) => sum + (a.timeTaken || 0), 0) / totalAttempts

  return {
    totalAttempts,
    correctRate,
    avgTime: Math.round(avgTime / 1000)
  }
})

const partialCredit = computed(() => {
  if (!isFillBlank.value || !submitted.value) return 0

  const userAns = textAnswer.value.trim().toLowerCase()
  const correctAnswers = props.acceptedAnswers || []

  let bestScore = 0
  for (const correct of correctAnswers) {
    const similarity = calculateSimilarity(userAns, correct.toLowerCase())
    bestScore = Math.max(bestScore, similarity)
  }

  if (bestScore >= 0.9) return 100
  if (bestScore >= 0.7) return 75
  if (bestScore >= 0.5) return 50
  if (bestScore >= 0.3) return 25
  return 0
})

const checkFillBlankAnswer = (): boolean => {
  const userAns = props.caseSensitive ? textAnswer.value.trim() : textAnswer.value.trim().toLowerCase()
  const correctAnswers = props.acceptedAnswers || []

  return correctAnswers.some(ans => {
    const target = props.caseSensitive ? ans : ans.toLowerCase()
    return userAns === target || calculateSimilarity(userAns, target) >= 0.9
  })
}

const calculateSimilarity = (str1: string, str2: string): number => {
  const len1 = str1.length
  const len2 = str2.length
  const matrix: number[][] = []

  for (let i = 0; i <= len1; i++) {
    matrix[i] = [i]
  }
  for (let j = 0; j <= len2; j++) {
    matrix[0][j] = j
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (str1[i - 1] === str2[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1]
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        )
      }
    }
  }

  const distance = matrix[len1][len2]
  const maxLen = Math.max(len1, len2)
  return maxLen === 0 ? 1 : (maxLen - distance) / maxLen
}

const select = (index: number) => {
  if (submitted.value) return
  selectedIndex.value = index
}

const revealHint = (index: number) => {
  if (index < hintsRevealed.value || !props.hints) return
  hintsRevealed.value = index + 1
  emit('hint-used', index)
}

const submit = () => {
  if (!canSubmit.value) return

  const timeTaken = Date.now() - startTime.value
  submitted.value = true

  const attemptData = {
    correct: isCorrect.value,
    hintsUsed: hintsRevealed.value
  }

  emit('complete', isCorrect.value)
  emit('attempt', attemptData)

  // Save attempt to localStorage
  const attempt: AttemptRecord = {
    question: props.question,
    questionId: questionId.value,
    correct: isCorrect.value,
    hintsUsed: hintsRevealed.value,
    timestamp: Date.now(),
    timeTaken
  }

  if (isFillBlank.value) {
    attempt.textAnswer = textAnswer.value
  }

  const attempts = JSON.parse(localStorage.getItem('quizAttempts') || '[]') as AttemptRecord[]
  attempts.push(attempt)
  localStorage.setItem('quizAttempts', JSON.stringify(attempts))

  // Also save to legacy quizResults for backwards compatibility
  if (props.tutorialId) {
    const quizResults = JSON.parse(localStorage.getItem('quizResults') || '{}')
    quizResults[props.tutorialId] = quizResults[props.tutorialId] || []
    quizResults[props.tutorialId].push({
      question: props.question,
      correct: isCorrect.value,
      timestamp: Date.now()
    })
    localStorage.setItem('quizResults', JSON.stringify(quizResults))

    // Dispatch custom event for ProgressTracker
    const event = new CustomEvent('quiz-completed', {
      detail: {
        tutorialId: props.tutorialId,
        correct: isCorrect.value,
        total: props.options.length
      }
    })
    window.dispatchEvent(event)
  }
}

const reset = () => {
  selectedIndex.value = null
  textAnswer.value = ''
  submitted.value = false
  hintsRevealed.value = 0
  startTime.value = Date.now()
}

const toggleHistory = () => {
  showHistory.value = !showHistory.value
}

const formatDate = (timestamp: number): string => {
  const date = new Date(timestamp)
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

onMounted(() => {
  startTime.value = Date.now()
})
</script>

<template>
  <div class="quiz-container">
    <div class="quiz-header">
      <h4>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        Knowledge Check
        <span v-if="type !== 'multiple-choice'" class="quiz-type-badge">{{ type === 'true-false' ? 'T/F' : 'Fill In' }}</span>
      </h4>
      <button v-if="stats" class="history-toggle" @click="toggleHistory" :title="showHistory ? 'Hide history' : 'Show history'">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
      </button>
    </div>

    <div class="quiz-body">
      <!-- Stats Display -->
      <div v-if="stats && showHistory" class="quiz-stats">
        <div class="stat-item">
          <span class="stat-label">Attempts:</span>
          <span class="stat-value">{{ stats.totalAttempts }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Success Rate:</span>
          <span class="stat-value">{{ stats.correctRate }}%</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Avg Time:</span>
          <span class="stat-value">{{ stats.avgTime }}s</span>
        </div>
      </div>

      <!-- Attempt History -->
      <div v-if="showHistory && attemptHistory.length > 0" class="attempt-history">
        <h5>Recent Attempts</h5>
        <div class="history-list">
          <div v-for="(attempt, idx) in attemptHistory" :key="idx" class="history-item" :class="{ correct: attempt.correct }">
            <div class="history-icon">
              <svg v-if="attempt.correct" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </div>
            <div class="history-details">
              <div class="history-meta">
                <span class="history-date">{{ formatDate(attempt.timestamp) }}</span>
                <span v-if="attempt.hintsUsed > 0" class="history-hints">{{ attempt.hintsUsed }} hint(s)</span>
              </div>
              <div v-if="attempt.textAnswer" class="history-answer">Answer: "{{ attempt.textAnswer }}"</div>
            </div>
          </div>
        </div>
      </div>

      <p class="quiz-question">{{ question }}</p>

      <!-- Hints Section -->
      <div v-if="hints && hints.length > 0 && !submitted" class="hints-section">
        <div v-for="(hint, index) in hints" :key="index" class="hint-item">
          <button
            v-if="index >= hintsRevealed"
            class="hint-btn"
            @click="revealHint(index)"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="16" x2="12" y2="12"/>
              <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            Show Hint {{ index + 1 }}
            <span v-if="hint.penalty" class="hint-penalty">(-{{ hint.penalty }}%)</span>
          </button>
          <div v-else class="hint-content">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            {{ hint.text }}
          </div>
        </div>
      </div>

      <!-- Fill-in-the-blank input -->
      <div v-if="isFillBlank" class="fill-blank-input">
        <input
          v-model="textAnswer"
          type="text"
          placeholder="Type your answer here..."
          :disabled="submitted"
          @keyup.enter="submit"
          class="text-input"
        />
      </div>

      <!-- Multiple choice / True-False options -->
      <div v-else class="quiz-options">
        <div
          v-for="(option, index) in options"
          :key="index"
          class="quiz-option"
          :class="{
            selected: selectedIndex === index && !submitted,
            correct: submitted && option.correct,
            incorrect: submitted && selectedIndex === index && !option.correct,
            'true-false': isTrueFalse
          }"
          @click="select(index)"
        >
          <span class="option-letter">{{ isTrueFalse ? (index === 0 ? 'T' : 'F') : String.fromCharCode(65 + index) }}</span>
          <span class="option-text">{{ option.text }}</span>
          <span v-if="submitted && option.correct" class="option-icon correct-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </span>
          <span v-if="submitted && selectedIndex === index && !option.correct" class="option-icon incorrect-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </span>
        </div>
      </div>

      <div class="quiz-actions" v-if="!submitted">
        <button
          class="quiz-submit"
          @click="submit"
          :disabled="!canSubmit"
          :title="!canSubmit ? 'Please select an answer' : ''"
        >
          Check Answer
        </button>
      </div>

      <div v-if="submitted" class="quiz-feedback" :class="{ correct: isCorrect, incorrect: !isCorrect, partial: partialCredit > 0 && !isCorrect }">
        <strong>
          <span v-if="isCorrect">Correct!</span>
          <span v-else-if="partialCredit > 0">Partial Credit: {{ partialCredit }}%</span>
          <span v-else>Not quite!</span>
        </strong>
        <p v-if="isFillBlank && !isCorrect && acceptedAnswers" class="correct-answers">
          Accepted answers: {{ acceptedAnswers.join(', ') }}
        </p>
        <p v-if="explanation">{{ explanation }}</p>
        <div v-if="hintsRevealed > 0" class="feedback-meta">
          <span>Hints used: {{ hintsRevealed }}</span>
        </div>
        <button class="quiz-retry" @click="reset">Try Again</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Header */
.quiz-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.quiz-type-badge {
  display: inline-block;
  padding: 2px 8px;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  margin-left: 8px;
}

.history-toggle {
  background: transparent;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--vp-c-text-2);
  transition: all 0.2s ease;
}

.history-toggle:hover {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

/* Stats */
.quiz-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 12px;
  padding: 12px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  margin-bottom: 16px;
  animation: slideDown 0.3s ease-out;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

/* Attempt History */
.attempt-history {
  margin-bottom: 16px;
  padding: 12px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  animation: slideDown 0.3s ease-out;
}

.attempt-history h5 {
  margin: 0 0 12px 0;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.history-item {
  display: flex;
  gap: 8px;
  padding: 8px;
  background: var(--vp-c-bg);
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  transition: all 0.2s ease;
}

.history-item.correct {
  border-color: rgba(76, 175, 80, 0.3);
  background: rgba(76, 175, 80, 0.05);
}

.history-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  flex-shrink: 0;
}

.history-item.correct .history-icon {
  background: rgba(76, 175, 80, 0.1);
  color: #4caf50;
}

.history-item:not(.correct) .history-icon {
  background: rgba(244, 67, 54, 0.1);
  color: #f44336;
}

.history-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.history-meta {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.history-hints {
  font-weight: 500;
}

.history-answer {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  font-style: italic;
}

/* Hints Section */
.hints-section {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hint-item {
  display: flex;
  align-items: center;
}

.hint-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
  transition: all 0.2s ease;
}

.hint-btn:hover {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg);
}

.hint-penalty {
  color: #f44336;
  font-weight: 500;
  font-size: 0.8rem;
}

.hint-content {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(62, 175, 124, 0.1);
  border: 1px solid rgba(62, 175, 124, 0.3);
  border-radius: 6px;
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
  animation: slideDown 0.3s ease-out;
}

.hint-content svg {
  color: var(--vp-c-brand-1);
  flex-shrink: 0;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Fill-in-the-blank */
.fill-blank-input {
  margin-bottom: 16px;
}

.text-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 1rem;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  transition: all 0.2s ease;
  font-family: inherit;
}

.text-input:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px rgba(62, 175, 124, 0.1);
}

.text-input:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  background: var(--vp-c-bg-soft);
}

/* Options */
.option-letter {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--vp-c-bg-soft);
  font-weight: 600;
  font-size: 0.875rem;
  margin-right: 12px;
  flex-shrink: 0;
  transition: all 0.2s ease;
}

.quiz-option.true-false .option-letter {
  width: 36px;
  height: 36px;
  font-size: 1rem;
}

.option-text {
  flex: 1;
}

.option-icon {
  margin-left: auto;
  display: flex;
  align-items: center;
}

.correct-icon {
  color: #4caf50;
}

.incorrect-icon {
  color: #f44336;
}

/* Actions */
.quiz-actions {
  margin-top: 16px;
}

.quiz-submit {
  padding: 10px 20px;
  background: var(--vp-c-brand-1);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.quiz-submit:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(62, 175, 124, 0.3);
}

.quiz-submit:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Feedback */
.quiz-feedback {
  margin-top: 16px;
  padding: 16px;
  border-radius: 8px;
  animation: slideDown 0.3s ease-out;
}

.quiz-feedback.partial {
  background: rgba(255, 167, 38, 0.1);
  border: 1px solid #ffa726;
  color: #f57c00;
}

.quiz-feedback p {
  margin: 8px 0;
}

.correct-answers {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  font-style: italic;
}

.feedback-meta {
  display: flex;
  gap: 16px;
  margin: 8px 0;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.quiz-retry {
  margin-top: 8px;
  padding: 6px 12px;
  background: transparent;
  border: 1px solid currentColor;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.quiz-retry:hover {
  background: var(--vp-c-bg-soft);
}

/* Responsive */
@media (max-width: 640px) {
  .quiz-stats {
    grid-template-columns: repeat(2, 1fr);
  }

  .history-meta {
    font-size: 0.7rem;
  }
}

/* Dark mode support */
.dark .history-item.correct {
  border-color: var(--color-success-border);
  background: var(--color-success-soft);
}

.dark .history-item.correct .history-icon {
  background: var(--color-success-soft);
}

.dark .history-item:not(.correct) .history-icon {
  background: var(--color-error-soft);
}

.dark .quiz-feedback.correct {
  background: var(--color-success-soft);
  border-color: var(--color-success);
  color: var(--color-success);
}

.dark .quiz-feedback.incorrect {
  background: var(--color-error-soft);
  border-color: var(--color-error);
  color: var(--color-error);
}

.dark .quiz-feedback.partial {
  background: var(--color-warning-soft);
  border-color: var(--color-warning);
  color: var(--color-warning);
}

.dark .hint-content {
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.4);
}

.dark .text-input {
  background: var(--vp-c-bg-alt);
}

.dark .quiz-stats {
  background: var(--vp-c-bg-alt);
}

.dark .attempt-history {
  background: var(--vp-c-bg-alt);
}
</style>
