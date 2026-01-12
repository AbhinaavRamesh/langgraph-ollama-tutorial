<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const props = defineProps<{
  code: string
  title?: string
  readonly?: boolean
}>()

const emit = defineEmits<{
  (e: 'run', code: string): void
}>()

const editorCode = ref(props.code)
const output = ref('')
const isRunning = ref(false)
const pyodideReady = ref(false)
const pyodide = ref<any>(null)

// Load Pyodide
onMounted(async () => {
  try {
    // @ts-ignore
    if (!window.loadPyodide) {
      const script = document.createElement('script')
      script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js'
      script.async = true
      document.head.appendChild(script)
      await new Promise(resolve => script.onload = resolve)
    }

    output.value = 'Loading Python runtime...'
    // @ts-ignore
    pyodide.value = await window.loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/'
    })

    // Pre-install common packages
    await pyodide.value.loadPackage(['micropip'])

    pyodideReady.value = true
    output.value = 'Python ready! Click "Run" to execute code.'
  } catch (error) {
    output.value = `Error loading Python: ${error}`
  }
})

watch(() => props.code, (newCode) => {
  editorCode.value = newCode
})

const runCode = async () => {
  if (!pyodideReady.value) {
    output.value = 'Python is still loading...'
    return
  }

  isRunning.value = true
  output.value = ''

  try {
    // Capture stdout
    pyodide.value.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
    `)

    // Run user code
    const result = await pyodide.value.runPythonAsync(editorCode.value)

    // Get stdout content
    const stdout = pyodide.value.runPython(`sys.stdout.getvalue()`)

    if (stdout) {
      output.value = stdout
    } else if (result !== undefined && result !== null) {
      output.value = String(result)
    } else {
      output.value = 'Code executed successfully (no output)'
    }

    emit('run', editorCode.value)
  } catch (error: any) {
    output.value = `Error: ${error.message || error}`
  } finally {
    isRunning.value = false
    // Reset stdout
    pyodide.value?.runPython(`sys.stdout = sys.__stdout__`)
  }
}

const resetCode = () => {
  editorCode.value = props.code
  output.value = ''
}

const copyCode = async () => {
  try {
    await navigator.clipboard.writeText(editorCode.value)
    // Could add a toast notification here
  } catch (error) {
    console.error('Failed to copy:', error)
  }
}
</script>

<template>
  <div class="code-playground">
    <div class="playground-header">
      <h4>
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        {{ title || 'Python Playground' }}
      </h4>
      <div class="playground-actions">
        <button class="playground-btn" @click="copyCode" title="Copy code">
          Copy
        </button>
        <button class="playground-btn" @click="resetCode" title="Reset to original">
          Reset
        </button>
        <button
          class="playground-btn run"
          @click="runCode"
          :disabled="!pyodideReady || isRunning"
        >
          {{ isRunning ? 'Running...' : 'Run' }}
        </button>
      </div>
    </div>

    <div class="playground-editor">
      <textarea
        v-model="editorCode"
        :readonly="readonly"
        spellcheck="false"
        class="editor-textarea"
      />
    </div>

    <div class="playground-output" :class="{ loading: isRunning }">
      <template v-if="isRunning">
        <span class="spinner"></span> Executing...
      </template>
      <template v-else>
        <pre>{{ output }}</pre>
      </template>
    </div>
  </div>
</template>

<style scoped>
.editor-textarea {
  width: 100%;
  min-height: 200px;
  padding: 16px;
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  line-height: 1.6;
  border: none;
  resize: vertical;
  background: var(--vp-c-bg-alt, #1e1e1e);
  color: var(--vp-c-text-1, #d4d4d4);
}

.editor-textarea:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
}

.playground-output pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--vp-c-divider);
  border-top-color: var(--vp-c-brand-1);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Dark mode support */
.dark .editor-textarea {
  background: #0d1117;
  color: #e6edf3;
}

.dark .playground-output {
  background: #0d1117;
  color: #e6edf3;
}
</style>
