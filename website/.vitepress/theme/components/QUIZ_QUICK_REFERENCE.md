# Quiz Component - Quick Reference

## Basic Usage

```vue
<Quiz
  question="Your question here?"
  :options="[
    { text: 'Option 1', correct: false },
    { text: 'Option 2', correct: true },
    { text: 'Option 3', correct: false }
  ]"
  explanation="Explanation shown after submission"
  tutorial-id="unique-id"
/>
```

## Question Types

### Multiple Choice (Default)
```vue
<Quiz
  question="Which is correct?"
  :options="[
    { text: 'Option A', correct: false },
    { text: 'Option B', correct: true },
    { text: 'Option C', correct: false },
    { text: 'Option D', correct: false }
  ]"
/>
```

### True/False
```vue
<Quiz
  type="true-false"
  question="Is this true?"
  :options="[
    { text: 'True', correct: true },
    { text: 'False', correct: false }
  ]"
/>
```

### Fill-in-the-Blank
```vue
<Quiz
  type="fill-blank"
  question="The answer is _____."
  :options="[]"
  :accepted-answers="['correct', 'right', 'accurate']"
  :case-sensitive="false"
/>
```

## Add Hints

```vue
<Quiz
  question="..."
  :options="[...]"
  :hints="[
    { text: 'First hint', penalty: 5 },
    { text: 'Second hint', penalty: 10 },
    { text: 'Final hint', penalty: 15 }
  ]"
/>
```

## Event Handlers

```vue
<Quiz
  question="..."
  :options="[...]"
  @attempt="onAttempt"
  @complete="onComplete"
  @hint-used="onHintUsed"
/>
```

```typescript
const onAttempt = (data: { correct: boolean, confidence: number, hintsUsed: number }) => {
  console.log('Attempt:', data)
}

const onComplete = (correct: boolean) => {
  console.log('Complete:', correct)
}

const onHintUsed = (hintIndex: number) => {
  console.log('Hint used:', hintIndex)
}
```

## Props Reference

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `question` | `string` | **Required** | Question text |
| `options` | `QuizOption[]` | **Required** | Answer options |
| `type` | `string` | `'multiple-choice'` | Question type |
| `explanation` | `string` | - | Post-answer explanation |
| `tutorialId` | `string` | - | ID for progress tracking |
| `hints` | `Hint[]` | - | Progressive hints |
| `acceptedAnswers` | `string[]` | - | Valid answers (fill-blank) |
| `caseSensitive` | `boolean` | `false` | Case sensitivity (fill-blank) |

## Features

### Automatic
- ✅ Confidence level tracking (required)
- ✅ Attempt history (localStorage)
- ✅ Time tracking
- ✅ Statistics calculation

### Optional
- 🔧 Progressive hints
- 🔧 Multiple question types
- 🔧 Partial credit (fill-blank)
- 🔧 Event emissions

### User Actions
- 👆 Select confidence (1-5 stars)
- 👆 Reveal hints (optional)
- 👆 Submit answer
- 👆 View history (click clock icon)
- 👆 Try again

## LocalStorage

### Key: `quizAttempts`
```json
[
  {
    "questionId": "tutorial-id-Question-text...",
    "question": "Full question text",
    "correct": true,
    "confidence": 4,
    "hintsUsed": 1,
    "timestamp": 1704067200000,
    "timeTaken": 15000,
    "textAnswer": "user answer"
  }
]
```

### Access Data
```javascript
// Get all attempts
const attempts = JSON.parse(localStorage.getItem('quizAttempts') || '[]')

// Clear history
localStorage.removeItem('quizAttempts')

// Get attempts for specific question
const filtered = attempts.filter(a => a.questionId === 'your-id')
```

## Partial Credit (Fill-in-the-Blank)

| Similarity | Credit |
|------------|--------|
| 90-100% | 100% |
| 70-89% | 75% |
| 50-69% | 50% |
| 30-49% | 25% |
| 0-29% | 0% |

## Styling Hooks

### CSS Classes
```css
/* Main container */
.quiz-container { }

/* Header with type badge */
.quiz-header { }
.quiz-type-badge { }
.history-toggle { }

/* Stats panel */
.quiz-stats { }
.stat-item { }

/* History */
.attempt-history { }
.history-item { }
.history-item.correct { }

/* Confidence */
.confidence-section { }
.star-btn { }
.star-btn.filled { }

/* Hints */
.hints-section { }
.hint-btn { }
.hint-content { }

/* Fill-blank */
.text-input { }

/* Feedback */
.quiz-feedback.correct { }
.quiz-feedback.incorrect { }
.quiz-feedback.partial { }
```

### CSS Variables
Uses VitePress theme variables:
- `--vp-c-brand-1` (primary color)
- `--vp-c-brand-2` (hover color)
- `--vp-c-bg` (background)
- `--vp-c-bg-soft` (soft background)
- `--vp-c-divider` (borders)
- `--vp-c-text-1` (primary text)
- `--vp-c-text-2` (secondary text)

## Common Patterns

### Tutorial End Quiz
```vue
<Quiz
  question="What did you learn?"
  :options="[...]"
  explanation="Great job!"
  tutorial-id="tutorial-01"
  @complete="markTutorialComplete"
/>
```

### Progressive Difficulty
```vue
<!-- Easy: Multiple choice -->
<Quiz type="multiple-choice" :options="[...]" />

<!-- Medium: True/False -->
<Quiz type="true-false" :options="[...]" />

<!-- Hard: Fill-blank -->
<Quiz type="fill-blank" :accepted-answers="[...]" />
```

### With Detailed Hints
```vue
<Quiz
  :hints="[
    { text: 'General direction', penalty: 5 },
    { text: 'More specific', penalty: 10 },
    { text: 'Almost the answer', penalty: 15 }
  ]"
/>
```

## Troubleshooting

### Quiz not showing
- Check Vue component registration
- Verify props are reactive (use `:options` not `options`)
- Check browser console for errors

### History not saving
- Ensure `tutorial-id` prop is set
- Check localStorage quota
- Verify browser allows localStorage

### Fuzzy matching too strict/loose
- Adjust `acceptedAnswers` array
- Add more variations
- Consider case-sensitivity

### Stars not clickable
- Check z-index conflicts
- Verify pointer-events not disabled
- Test on different browsers

## Best Practices

1. **Question Writing**
   - Be clear and concise
   - Avoid ambiguity
   - One correct answer

2. **Hints**
   - Start general, get specific
   - Don't give away the answer
   - Use reasonable penalties (5-15%)

3. **Fill-blank**
   - List common variations
   - Include plurals if applicable
   - Consider abbreviations

4. **Tutorial IDs**
   - Use consistent naming
   - Include section/topic
   - Make them unique

5. **Explanations**
   - Explain why it's correct
   - Reference key concepts
   - Link to more info

## Examples Repository

See full examples in:
- `/tutorials/quiz-demo.md` - Interactive demo
- `QuizExamples.md` - Code examples
- `QUIZ_ENHANCEMENT_SUMMARY.md` - Complete documentation
