import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import './custom.css'

// Import custom components
import CodePlayground from './components/CodePlayground.vue'
import Quiz from './components/Quiz.vue'
import ProgressTracker from './components/ProgressTracker.vue'
import TutorialCard from './components/TutorialCard.vue'
import PhaseProgress from './components/PhaseProgress.vue'
import QuizDashboard from './components/QuizDashboard.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register global components
    app.component('CodePlayground', CodePlayground)
    app.component('Quiz', Quiz)
    app.component('ProgressTracker', ProgressTracker)
    app.component('TutorialCard', TutorialCard)
    app.component('PhaseProgress', PhaseProgress)
    app.component('QuizDashboard', QuizDashboard)
  }
} satisfies Theme
