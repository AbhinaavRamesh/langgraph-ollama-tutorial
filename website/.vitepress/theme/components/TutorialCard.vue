<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

const props = defineProps<{
  number: number
  title: string
  description: string
  link: string
  phase: 'core' | 'rag' | 'multi-agent' | 'advanced'
}>()

const isCompleted = ref(false)

const tutorialId = computed(() => {
  const match = props.link.match(/(\d{2}-[\w-]+)/)
  return match ? match[1] : null
})

onMounted(() => {
  const progress = JSON.parse(localStorage.getItem('tutorialProgress') || '{}')
  if (tutorialId.value && progress[tutorialId.value]?.completed) {
    isCompleted.value = true
  }
})

const phaseLabel = computed(() => {
  const labels = {
    core: 'Core Patterns',
    rag: 'RAG Patterns',
    'multi-agent': 'Multi-Agent',
    advanced: 'Advanced'
  }
  return labels[props.phase]
})
</script>

<template>
  <a :href="link" class="tutorial-card" :class="{ completed: isCompleted }">
    <div class="card-header">
      <span class="number">Tutorial {{ String(number).padStart(2, '0') }}</span>
      <span v-if="isCompleted" class="completion-badge">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </span>
    </div>
    <h3>{{ title }}</h3>
    <p>{{ description }}</p>
    <span :class="['phase-badge', phase]">{{ phaseLabel }}</span>
  </a>
</template>

<style scoped>
.tutorial-card {
  text-decoration: none;
  color: inherit;
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.completion-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  background: #4caf50;
  border-radius: 50%;
  color: white;
}

.phase-badge {
  margin-top: auto;
  align-self: flex-start;
}
</style>
