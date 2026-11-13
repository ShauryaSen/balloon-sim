<!--
  The four kinds of realism check, each as a small picture (ERA5 grey, generated blue), the question it asks,
  and why a balloon cares. Adapted from the talk-fable draft.
-->
<template>
  <div class="checks">
    <div class="card">
      <svg viewBox="0 0 160 90">
        <path d="M8 80 C 40 80, 45 14, 70 14 S 105 80, 152 80" fill="none" stroke="#b7bcc6" stroke-width="3" />
        <path d="M8 80 C 50 80, 58 24, 82 24 S 118 80, 152 80" fill="none" stroke="#4f7db3" stroke-width="3" />
        <line x1="8" y1="82" x2="152" y2="82" stroke="#1a1a1a" stroke-width="1.2" />
      </svg>
      <h3>Wind speeds</h3>
      <p>Distribution of wind speeds, including extremes</p>
    </div>
    <div class="card" v-click>
      <svg viewBox="0 0 160 90">
        <path d="M10 14 C 50 22, 90 40, 150 74" fill="none" stroke="#b7bcc6" stroke-width="3" />
        <path d="M10 14 C 50 22, 80 44, 110 84" fill="none" stroke="#4f7db3" stroke-width="3" />
        <line x1="8" y1="8" x2="8" y2="84" stroke="#1a1a1a" stroke-width="1.2" /><line x1="8" y1="84" x2="152" y2="84" stroke="#1a1a1a" stroke-width="1.2" />
        <text x="150" y="20" fill="#888" text-anchor="end" style="font-size: 9px">large → small features</text>
      </svg>
      <h3>Structure in space</h3>
      <p>Energy at each spatial scale, and effective resolution</p>
    </div>
    <div class="card" v-click>
      <svg viewBox="0 0 160 90">
        <g stroke-width="3" stroke-linecap="round" fill="none">
          <g stroke="#b7bcc6"><path d="M50 16 h48 m-8 -5 l8 5 l-8 5" /><path d="M50 36 h22 m-8 -5 l8 5 l-8 5" /><path d="M70 56 h-26 m8 -5 l-8 5 l8 5" /><path d="M70 76 h-44 m8 -5 l-8 5 l8 5" /></g>
          <g stroke="#4f7db3" transform="translate(52,0)"><path d="M50 16 h40 m-8 -5 l8 5 l-8 5" /><path d="M50 36 h30 m-8 -5 l8 5 l-8 5" /><path d="M50 56 h18 m-8 -5 l8 5 l-8 5" /><path d="M50 76 h8 m-6 -5 l6 5 l-6 5" /></g>
        </g>
        <line x1="20" y1="8" x2="20" y2="84" stroke="#1a1a1a" stroke-width="1.2" />
      </svg>
      <h3>Change with altitude</h3>
      <p>Wind shear between neighbouring altitude levels</p>
    </div>
    <div class="card" v-click>
      <svg viewBox="0 0 160 90">
        <g fill="none" stroke-width="2.4" stroke-linecap="round">
          <g stroke="#b7bcc6"><path d="M14 46 C 60 40, 100 20, 150 10" /><path d="M14 46 C 60 46, 100 44, 150 40" /><path d="M14 46 C 60 52, 100 70, 150 80" /></g>
          <g stroke="#4f7db3"><path d="M14 46 C 60 44, 100 38, 150 34" /><path d="M14 46 C 60 47, 100 48, 150 50" /><path d="M14 46 C 60 49, 100 56, 150 62" /></g>
        </g>
        <circle cx="14" cy="46" r="4" fill="#1a1a1a" />
      </svg>
      <h3>Change over time</h3>
      <p>Energy at each time scale, and how fast drifting particles spread apart</p>
    </div>
  </div>
</template>

<style scoped>
.checks { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 0.8rem; }
.card { border: 1px solid #e3e3e3; border-radius: 6px; padding: 12px 14px 14px; }
.card svg { width: 100%; height: 80px; margin-bottom: 6px; }
.card h3 { font-size: 1.05rem; font-weight: 600; margin: 0 0 4px; }
.card p { font-size: 0.85rem !important; line-height: 1.35 !important; margin: 0 0 6px; color: #333; }
.card p.why { color: #4f7db3 !important; }
</style>
