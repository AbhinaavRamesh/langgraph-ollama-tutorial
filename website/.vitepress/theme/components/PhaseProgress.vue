<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

const props = defineProps<{
  phase: 'core' | 'rag' | 'multi-agent' | 'advanced'
  tutorials: string[]
}>()

const completedCount = ref(0)

const percentage = computed(() =>
  Math.round((completedCount.value / props.tutorials.length) * 100)
)

const phaseInfo = computed(() => {
  const info = {
    core: { name: 'Core Patterns', color: '#1565c0', bg: '#e3f2fd' },
    rag: { name: 'RAG Patterns', color: '#ef6c00', bg: '#fff3e0' },
    'multi-agent': { name: 'Multi-Agent Patterns', color: '#7b1fa2', bg: '#f3e5f5' },
    advanced: { name: 'Advanced Reasoning', color: '#c2185b', bg: '#fce4ec' }
  }
  return info[props.phase]
})

onMounted(() => {
  const progress = JSON.parse(localStorage.getItem('tutorialProgress') || '{}')
  completedCount.value = props.tutorials.filter(t => {
    const id = t.match(/(\d{2}-[\w-]+)/)?.[1]
    return id && progress[id]?.completed
  }).length
})
</script>

<template>
  <div class="phase-progress" :style="{ borderColor: phaseInfo.color }">
    <div class="phase-header">
      <h3 :style="{ color: phaseInfo.color }">{{ phaseInfo.name }}</h3>
      <span class="phase-stats">{{ completedCount }}/{{ tutorials.length }} completed</span>
    </div>

    <div class="phase-progress-bar">
      <div
        class="phase-progress-fill"
        :style="{
          width: `${percentage}%`,
          background: `linear-gradient(90deg, ${phaseInfo.color}, ${phaseInfo.color}dd)`
        }"
      ></div>
    </div>

    <div v-if="percentage === 100" class="phase-complete">
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
      Phase Complete!
    </div>
  </div>
</template>

<style scoped>
.phase-progress {
  padding: 20px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border-left: 4px solid;
  margin: 24px 0;
}

.phase-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.phase-header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.phase-stats {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.phase-progress-bar {
  height: 8px;
  background: var(--vp-c-divider);
  border-radius: 4px;
  overflow: hidden;
}

.phase-progress-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}

.phase-complete {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  color: #4caf50;
  font-weight: 500;
}
</style>
