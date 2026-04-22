import { defineConfig } from 'unocss'

export default defineConfig({
  shortcuts: {
    // ── Type Scale (Major Third 1.25×) ──────────────
    'type-display':  'text-4xl font-bold tracking-tight',
    'type-title':    'text-3xl font-bold',
    'type-headline': 'text-2xl font-semibold',
    'type-subhead':  'text-xl font-medium',
    'type-body':     'text-base leading-relaxed',
    'type-callout':  'text-lg font-medium',
    'type-caption':  'text-sm opacity-70',
    'type-footnote': 'text-xs opacity-50',

    // ── Cards ───────────────────────────────────────
    'card-blue': 'p-4 bg-blue-50 rounded-lg '
      + 'dark:bg-blue-900/30 border border-blue-200/50 '
      + 'dark:border-blue-700/30',
    'card-green': 'p-4 bg-green-50 rounded-lg '
      + 'dark:bg-green-900/30 border border-green-200/50 '
      + 'dark:border-green-700/30',
    'card-amber': 'p-4 bg-amber-50 rounded-lg '
      + 'dark:bg-amber-900/30 border border-amber-200/50 '
      + 'dark:border-amber-700/30',
    'card-red': 'p-4 bg-red-50 rounded-lg '
      + 'dark:bg-red-900/30 border border-red-200/50 '
      + 'dark:border-red-700/30',
  },
})
