# Quiz Component Architecture

## Component Structure

```
Quiz.vue
├── Script Section (TypeScript)
│   ├── Interfaces
│   │   ├── QuizOption
│   │   ├── Hint
│   │   └── AttemptRecord
│   │
│   ├── Props (withDefaults)
│   │   ├── question: string (required)
│   │   ├── options: QuizOption[] (required)
│   │   ├── type: 'multiple-choice' | 'true-false' | 'fill-blank'
│   │   ├── hints: Hint[]
│   │   ├── acceptedAnswers: string[]
│   │   ├── caseSensitive: boolean
│   │   ├── explanation: string
│   │   └── tutorialId: string
│   │
│   ├── Reactive State (ref)
│   │   ├── selectedIndex: number | null
│   │   ├── textAnswer: string
│   │   ├── submitted: boolean
│   │   ├── confidence: number (0-5)
│   │   ├── hintsRevealed: number
│   │   ├── startTime: number
│   │   └── showHistory: boolean
│   │
│   ├── Computed Properties
│   │   ├── questionId: string (unique identifier)
│   │   ├── correctIndex: number
│   │   ├── isCorrect: boolean
│   │   ├── isTrueFalse: boolean
│   │   ├── isFillBlank: boolean
│   │   ├── canSubmit: boolean
│   │   ├── attemptHistory: AttemptRecord[]
│   │   ├── stats: object (totals, rates, averages)
│   │   └── partialCredit: number (0-100)
│   │
│   ├── Methods
│   │   ├── select(index)
│   │   ├── setConfidence(level)
│   │   ├── revealHint(index)
│   │   ├── submit()
│   │   ├── reset()
│   │   ├── toggleHistory()
│   │   ├── formatDate(timestamp)
│   │   ├── checkFillBlankAnswer()
│   │   └── calculateSimilarity(str1, str2)
│   │
│   └── Lifecycle
│       └── onMounted() - Initialize startTime
│
├── Template Section (Vue)
│   ├── Container (.quiz-container)
│   │   │
│   │   ├── Header (.quiz-header)
│   │   │   ├── Title + Icon
│   │   │   ├── Type Badge (if not multiple-choice)
│   │   │   └── History Toggle Button
│   │   │
│   │   └── Body (.quiz-body)
│   │       │
│   │       ├── Stats Display (v-if stats && showHistory)
│   │       │   └── Grid of 4 stats
│   │       │
│   │       ├── Attempt History (v-if showHistory)
│   │       │   └── List of recent attempts
│   │       │
│   │       ├── Question Text
│   │       │
│   │       ├── Confidence Section (v-if !submitted)
│   │       │   └── 5 Star Buttons
│   │       │
│   │       ├── Hints Section (v-if hints && !submitted)
│   │       │   └── Sequential hint buttons/content
│   │       │
│   │       ├── Answer Input (conditional)
│   │       │   ├── Fill-blank: Text Input
│   │       │   └── MC/TF: Option Buttons
│   │       │
│   │       ├── Actions (v-if !submitted)
│   │       │   └── Submit Button
│   │       │
│   │       └── Feedback (v-if submitted)
│   │           ├── Result Message
│   │           ├── Correct Answers (fill-blank)
│   │           ├── Explanation
│   │           ├── Metadata (confidence, hints)
│   │           └── Retry Button
│   │
│   └── Style Section (Scoped CSS)
│       ├── Header Styles
│       ├── Stats Styles
│       ├── History Styles
│       ├── Confidence Styles
│       ├── Hints Styles
│       ├── Fill-blank Styles
│       ├── Options Styles
│       ├── Actions Styles
│       ├── Feedback Styles
│       ├── Animations
│       └── Responsive Rules
│
└── Emits
    ├── @attempt(data)
    ├── @complete(correct)
    └── @hint-used(hintIndex)
```

## Data Flow

```
User Interaction Flow:
┌─────────────────────────────────────────────────────────────┐
│                     Component Mounted                        │
│                    startTime = now()                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              User Selects Confidence (Required)              │
│                  setConfidence(1-5)                          │
│                  confidence.value = n                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            User May Reveal Hints (Optional)                  │
│                  revealHint(index)                           │
│              hintsRevealed.value++                           │
│              emit('hint-used', index)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   User Answers Question                      │
│   ┌──────────────────┬──────────────────┬─────────────────┐ │
│   │ Multiple Choice  │   True/False     │  Fill-in-Blank  │ │
│   │ select(index)    │  select(index)   │ textAnswer.value│ │
│   └──────────────────┴──────────────────┴─────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   canSubmit Validation                       │
│      confidence > 0 && (answer selected OR text entered)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    User Clicks Submit                        │
│                       submit()                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 1. Calculate timeTaken = now() - startTime          │   │
│   │ 2. Set submitted = true                             │   │
│   │ 3. Evaluate answer (isCorrect)                      │   │
│   │ 4. Calculate partialCredit (fill-blank)            │   │
│   │ 5. Create AttemptRecord                             │   │
│   │ 6. Save to localStorage('quizAttempts')            │   │
│   │ 7. Save to localStorage('quizResults') [legacy]    │   │
│   │ 8. Dispatch custom event (ProgressTracker)         │   │
│   │ 9. emit('attempt', data)                            │   │
│   │ 10. emit('complete', correct)                       │   │
│   └─────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feedback Displayed                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ • Correct/Incorrect/Partial message                 │   │
│   │ • Explanation (if provided)                         │   │
│   │ • Metadata (confidence, hints used)                 │   │
│   │ • Correct answers (fill-blank if wrong)            │   │
│   │ • Retry button                                      │   │
│   └─────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             User May Click "Try Again"                       │
│                       reset()                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ selectedIndex = null                                │   │
│   │ textAnswer = ''                                     │   │
│   │ submitted = false                                   │   │
│   │ confidence = 0                                      │   │
│   │ hintsRevealed = 0                                   │   │
│   │ startTime = now()                                   │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## State Management

```
Reactive State Tree:
├── selectedIndex (number | null)
│   └── Used by: Multiple choice, True/False
│
├── textAnswer (string)
│   └── Used by: Fill-in-the-blank
│
├── submitted (boolean)
│   ├── Controls: Feedback visibility
│   ├── Controls: Input enabled/disabled
│   └── Controls: Action button visibility
│
├── confidence (number 0-5)
│   ├── Required: Before submission
│   ├── Tracked: In attempt history
│   └── Displayed: In feedback metadata
│
├── hintsRevealed (number)
│   ├── Controls: Which hints visible
│   ├── Tracked: In attempt history
│   └── Displayed: In feedback metadata
│
├── startTime (number)
│   ├── Set: On mount and reset
│   ├── Used: To calculate timeTaken
│   └── Precision: Milliseconds
│
└── showHistory (boolean)
    ├── Controls: Stats panel visibility
    ├── Controls: Attempts list visibility
    └── Toggled: By clock icon button
```

## Computed Properties Flow

```
Computed Dependencies:
┌─────────────────────────────────────────────────────────┐
│ questionId                                              │
│   ← props.tutorialId                                    │
│   ← props.question                                      │
│   → Used by: attemptHistory filter                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ correctIndex                                            │
│   ← props.options                                       │
│   → Used by: isCorrect check (MC/TF)                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ isCorrect                                               │
│   ← props.type                                          │
│   ← selectedIndex OR textAnswer                        │
│   ← correctIndex (MC/TF)                               │
│   ← checkFillBlankAnswer() (fill-blank)               │
│   → Used by: Feedback display                          │
│   → Used by: Attempt recording                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ canSubmit                                               │
│   ← confidence > 0                                      │
│   ← selectedIndex !== null OR textAnswer.length > 0    │
│   → Used by: Submit button disabled state              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ attemptHistory                                          │
│   ← localStorage('quizAttempts')                       │
│   ← questionId                                          │
│   → Used by: stats calculation                         │
│   → Used by: History display                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ stats                                                   │
│   ← attemptHistory                                      │
│   → Calculates: total, rate, avg time, avg confidence  │
│   → Used by: Stats panel display                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ partialCredit                                           │
│   ← props.type === 'fill-blank'                        │
│   ← submitted === true                                  │
│   ← textAnswer                                          │
│   ← props.acceptedAnswers                              │
│   → Used by: Feedback styling and message              │
└─────────────────────────────────────────────────────────┘
```

## LocalStorage Integration

```
Storage Architecture:

┌─────────────────────────────────────────────────────────┐
│              Browser LocalStorage                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Key: 'quizAttempts'                                    │
│  Type: Array<AttemptRecord>                             │
│  ┌────────────────────────────────────────────────┐     │
│  │ [                                              │     │
│  │   {                                            │     │
│  │     questionId: "id-Question-text...",        │     │
│  │     question: "Full text",                    │     │
│  │     correct: true,                            │     │
│  │     confidence: 4,                            │     │
│  │     hintsUsed: 1,                             │     │
│  │     textAnswer: "answer" | undefined,         │     │
│  │     timestamp: 1704067200000,                 │     │
│  │     timeTaken: 15000                          │     │
│  │   },                                           │     │
│  │   ...                                          │     │
│  │ ]                                              │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  Key: 'quizResults' [Legacy]                            │
│  Type: Object<tutorialId, Array>                        │
│  ┌────────────────────────────────────────────────┐     │
│  │ {                                              │     │
│  │   "tutorial-01": [                            │     │
│  │     {                                          │     │
│  │       question: "...",                        │     │
│  │       correct: true,                          │     │
│  │       timestamp: 1704067200000                │     │
│  │     }                                          │     │
│  │   ]                                            │     │
│  │ }                                              │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
└─────────────────────────────────────────────────────────┘

Read Operations:
  • On mount: Load for stats/history display
  • On submit: Load to append new attempt

Write Operations:
  • On submit: Append to quizAttempts array
  • On submit: Append to quizResults object (if tutorialId)

Cleanup:
  • Manual via browser tools or:
    localStorage.removeItem('quizAttempts')
    localStorage.removeItem('quizResults')
```

## Event System

```
Event Emission Flow:

Component Internal State Changes
           │
           ▼
┌──────────────────────────┐
│  User reveals hint       │
│  revealHint(index)       │
└────────────┬─────────────┘
             │
             ▼
    emit('hint-used', index)
             │
             ▼
    Parent Component Handler
             │
             ▼
    Analytics/Tracking


Component Internal State Changes
           │
           ▼
┌──────────────────────────┐
│  User submits answer     │
│  submit()                │
└────────────┬─────────────┘
             │
             ├─► emit('attempt', {
             │     correct: boolean,
             │     confidence: number,
             │     hintsUsed: number
             │   })
             │         │
             │         ▼
             │   Parent Handler
             │         │
             │         ▼
             │   Update Progress
             │
             └─► emit('complete', correct)
                       │
                       ▼
                 Parent Handler
                       │
                       ▼
                 Legacy Support


Component Internal
           │
           ▼
┌──────────────────────────┐
│  window.dispatchEvent    │
│  CustomEvent             │
│  'quiz-completed'        │
└────────────┬─────────────┘
             │
             ▼
    ProgressTracker Component
             │
             ▼
    Global Progress Update
```

## Algorithm: Levenshtein Distance

```javascript
/**
 * Calculates similarity between two strings
 * Returns: 0.0 (completely different) to 1.0 (identical)
 */
calculateSimilarity(str1, str2) {
  // Build matrix
  const len1 = str1.length
  const len2 = str2.length
  const matrix = []

  // Initialize first column
  for (i = 0; i <= len1; i++) {
    matrix[i] = [i]
  }

  // Initialize first row
  for (j = 0; j <= len2; j++) {
    matrix[0][j] = j
  }

  // Fill matrix with edit distances
  for (i = 1; i <= len1; i++) {
    for (j = 1; j <= len2; j++) {
      if (str1[i-1] === str2[j-1]) {
        // Characters match, no edit needed
        matrix[i][j] = matrix[i-1][j-1]
      } else {
        // Take minimum of:
        // - Replace: matrix[i-1][j-1] + 1
        // - Insert:  matrix[i][j-1] + 1
        // - Delete:  matrix[i-1][j] + 1
        matrix[i][j] = Math.min(
          matrix[i-1][j-1] + 1,  // replace
          matrix[i][j-1] + 1,    // insert
          matrix[i-1][j] + 1     // delete
        )
      }
    }
  }

  // Convert distance to similarity
  distance = matrix[len1][len2]
  maxLength = Math.max(len1, len2)
  similarity = (maxLength - distance) / maxLength

  return similarity
}
```

## CSS Architecture

```
Style Organization:

Global (from custom.css)
  ↓ CSS Variables
  • --vp-c-brand-1
  • --vp-c-brand-2
  • --vp-c-bg
  • --vp-c-bg-soft
  • --vp-c-divider
  • --vp-c-text-1
  • --vp-c-text-2

Component Scoped Styles
  ├── Layout (.quiz-container, .quiz-header, .quiz-body)
  ├── Header (.quiz-type-badge, .history-toggle)
  ├── Stats (.quiz-stats, .stat-item)
  ├── History (.attempt-history, .history-item)
  ├── Confidence (.confidence-section, .star-btn)
  ├── Hints (.hints-section, .hint-btn, .hint-content)
  ├── Fill-blank (.fill-blank-input, .text-input)
  ├── Options (.quiz-option, .option-letter)
  ├── Actions (.quiz-submit)
  ├── Feedback (.quiz-feedback, .feedback-meta)
  └── Responsive (@media max-width: 640px)

Animations
  ├── @keyframes slideDown
  │   └── Used by: History, hints, feedback
  └── @keyframes starPop
      └── Used by: Confidence stars

States
  ├── .selected (option selected)
  ├── .correct (correct answer)
  ├── .incorrect (wrong answer)
  ├── .partial (partial credit)
  ├── .filled (star filled)
  ├── .active (star active)
  └── .true-false (TF variant)
```

## Performance Characteristics

```
Operation Complexity:

UI Rendering:
  • Initial render: O(n) where n = number of options
  • Re-render on state change: O(1) reactive updates

Computation:
  • questionId generation: O(1)
  • correctIndex lookup: O(n) where n = options
  • canSubmit check: O(1)
  • attemptHistory filter: O(m) where m = total attempts
  • stats calculation: O(k) where k = relevant attempts (<= 5)
  • partialCredit calc: O(p * s²) where p = accepted answers, s = string length
  • Levenshtein: O(m * n) where m,n = string lengths

Storage:
  • Read from localStorage: O(1) browser operation
  • Parse JSON: O(m) where m = total attempts
  • Stringify JSON: O(m)
  • Write to localStorage: O(1) browser operation

Memory:
  • Reactive refs: ~10 variables
  • Computed properties: ~9 cached values
  • Event listeners: ~20 (on buttons/inputs)
  • LocalStorage: ~200-300 bytes per attempt

Typical Performance:
  • Initial mount: <50ms
  • State update: <10ms
  • Submit processing: <100ms
  • History toggle: <50ms
  • Fuzzy match: <5ms (strings <100 chars)
```

This architecture enables:
1. **Modularity**: Clear separation of concerns
2. **Reactivity**: Efficient Vue 3 Composition API
3. **Extensibility**: Easy to add new question types
4. **Performance**: Optimized computed properties and minimal DOM manipulation
5. **Persistence**: Robust localStorage integration
6. **Testing**: Clear data flow for unit testing
