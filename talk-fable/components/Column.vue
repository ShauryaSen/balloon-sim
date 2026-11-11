<!-- One column of air: an arrow per altitude. `winds` lists them from the top down (+ east, - west, size is speed). -->
<script setup lang="ts">
defineProps<{ winds: number[]; colour?: string }>()
</script>

<template>
  <svg :viewBox="`0 0 220 150`" style="width: 100%; height: 150px">
    <line x1="110" y1="8" x2="110" y2="142" stroke="#e6e8ec" stroke-width="2" />
    <g v-for="(w, i) in winds" :key="i" :transform="`translate(110, ${winds.length === 1 ? 75 : 18 + i * (114 / (winds.length - 1))})`">
      <line x1="0" y1="0" :x2="90 * w" y2="0" :stroke="colour || '#1c2433'" stroke-width="3.2" stroke-linecap="round" />
      <path :d="`M ${90 * w - 10 * Math.sign(w)} -7 L ${90 * w} 0 L ${90 * w - 10 * Math.sign(w)} 7`" fill="none" :stroke="colour || '#1c2433'" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" />
      <circle r="3.5" :fill="colour || '#1c2433'" />
    </g>
  </svg>
</template>
