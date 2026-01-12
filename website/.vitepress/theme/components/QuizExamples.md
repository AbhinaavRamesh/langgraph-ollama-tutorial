# Quiz Component - Enhanced Features

This document demonstrates all the enhanced features of the Quiz component.

## Phase 1 Features

### Confidence Level Tracking

Users must select a confidence level (1-5 stars) before submitting their answer. This helps track learning progress and metacognition.

```vue
<Quiz
  question="What is the primary purpose of LangGraph?"
  :options="[
    { text: 'To create graphs', correct: false },
    { text: 'To build stateful AI workflows', correct: true },
    { text: 'To visualize data', correct: false },
    { text: 'To manage databases', correct: false }
  ]"
  explanation="LangGraph is designed to build stateful, multi-step AI workflows with cycles and persistence."
  tutorial-id="intro-to-langgraph"
/>
```

### Attempt History Tracking

All attempts are recorded in localStorage with:
- Timestamp
- Confidence level
- Hints used
- Text answer (for fill-in-the-blank)
- Time taken to complete

Click the history icon in the header to view:
- Total attempts
- Success rate percentage
- Average time taken
- Average confidence level
- List of recent attempts with details

## Phase 2 Features

### Multiple Question Types

#### 1. Multiple Choice (Default)

Standard multiple-choice questions with labeled options (A, B, C, D...).

```vue
<Quiz
  type="multiple-choice"
  question="Which component manages state in LangGraph?"
  :options="[
    { text: 'StateGraph', correct: true },
    { text: 'MessageGraph', correct: false },
    { text: 'WorkflowGraph', correct: false }
  ]"
/>
```

#### 2. True/False

Simplified two-option format with T/F labels.

```vue
<Quiz
  type="true-false"
  question="LangGraph supports conditional edges for dynamic routing."
  :options="[
    { text: 'True', correct: true },
    { text: 'False', correct: false }
  ]"
  explanation="Yes! Conditional edges allow dynamic routing based on state."
/>
```

#### 3. Fill-in-the-Blank

Text input with fuzzy matching and partial credit support.

```vue
<Quiz
  type="fill-blank"
  question="The _____ class is used to define stateful multi-agent workflows in LangGraph."
  :options="[]"
  :accepted-answers="['StateGraph', 'StatefulGraph', 'state graph']"
  :case-sensitive="false"
  explanation="StateGraph is the core class for building stateful workflows."
/>
```

**Partial Credit Scoring:**
- 90%+ similarity = 100% credit
- 70-89% similarity = 75% credit
- 50-69% similarity = 50% credit
- 30-49% similarity = 25% credit
- <30% similarity = 0% credit

### Progressive Hint System

Add hints that reveal progressively. Optionally include penalties for using hints.

```vue
<Quiz
  question="How do you add persistence to a LangGraph workflow?"
  :options="[
    { text: 'Use MemorySaver checkpointer', correct: true },
    { text: 'Add a database node', correct: false },
    { text: 'Enable auto-save', correct: false },
    { text: 'Use SessionStorage', correct: false }
  ]"
  :hints="[
    { text: 'Think about what saves state between runs...', penalty: 5 },
    { text: 'Look for a class that implements BaseCheckpointSaver', penalty: 10 }
  ]"
  explanation="MemorySaver is a checkpointer that provides in-memory persistence."
/>
```

### Events Emitted

The component now emits three events:

#### `@attempt`
Fired when an answer is submitted.

```typescript
{
  correct: boolean
  confidence: number // 1-5
  hintsUsed: number
}
```

#### `@complete`
Fired when quiz is completed (maintains backwards compatibility).

```typescript
correct: boolean
```

#### `@hint-used`
Fired when a hint is revealed.

```typescript
hintIndex: number // 0-based index
```

## Complete Example

Here's a comprehensive example using all features:

```vue
<Quiz
  type="fill-blank"
  question="In LangGraph, the _____ function is used to define conditional routing logic."
  :options="[]"
  :accepted-answers="['conditional_edge', 'add_conditional_edges', 'conditionalEdge']"
  :case-sensitive="false"
  :hints="[
    { text: 'It starts with "conditional"', penalty: 5 },
    { text: 'It relates to graph edges/connections', penalty: 10 }
  ]"
  explanation="The add_conditional_edges() function allows you to define routing logic based on the current state."
  tutorial-id="advanced-routing"
  @attempt="handleAttempt"
  @complete="handleComplete"
  @hint-used="handleHintUsed"
/>
```

```vue
<script setup>
const handleAttempt = (data) => {
  console.log('Attempt data:', data)
  // Track analytics, update progress, etc.
}

const handleComplete = (correct) => {
  console.log('Quiz completed:', correct)
}

const handleHintUsed = (hintIndex) => {
  console.log('Hint revealed:', hintIndex)
}
</script>
```

## Props Reference

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `question` | `string` | Required | The question text |
| `options` | `QuizOption[]` | Required | Array of answer options |
| `explanation` | `string` | `undefined` | Explanation shown after submission |
| `tutorialId` | `string` | `undefined` | ID for tracking progress |
| `type` | `'multiple-choice' \| 'true-false' \| 'fill-blank'` | `'multiple-choice'` | Question type |
| `hints` | `Hint[]` | `undefined` | Progressive hints |
| `acceptedAnswers` | `string[]` | `undefined` | Valid answers for fill-blank |
| `caseSensitive` | `boolean` | `false` | Case sensitivity for fill-blank |

## LocalStorage Structure

### quizAttempts
Array of all attempts across all quizzes:

```json
[
  {
    "question": "What is LangGraph?",
    "questionId": "intro-to-langgraph-What-is-LangGraph?",
    "correct": true,
    "confidence": 4,
    "hintsUsed": 0,
    "timestamp": 1704067200000,
    "timeTaken": 15000,
    "textAnswer": "StateGraph" // Only for fill-blank
  }
]
```

### quizResults (Legacy)
Maintained for backwards compatibility:

```json
{
  "intro-to-langgraph": [
    {
      "question": "What is LangGraph?",
      "correct": true,
      "timestamp": 1704067200000
    }
  ]
}
```

## Styling Customization

All styles use CSS custom properties for easy theming. Key classes:

- `.confidence-section` - Star rating container
- `.hints-section` - Hints display
- `.fill-blank-input` - Text input styling
- `.attempt-history` - History panel
- `.quiz-stats` - Statistics display
- `.history-item` - Individual attempt entry

Animations include:
- `slideDown` - For revealing elements
- `starPop` - For star selection feedback
