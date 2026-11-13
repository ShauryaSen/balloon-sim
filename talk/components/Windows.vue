<!--
  InfiniteDiffusion in one picture: a region is covered by overlapping windows, denoised together and blended
  where they overlap, so there are no seams. Purely schematic.
-->
<script setup lang="ts">
const SIZE = 110 // one window
const STEP = 80 // windows overlap by SIZE - STEP
const cols = [0, 1, 2]
const rows = [0, 1]
</script>

<template>
  <svg viewBox="-10 -10 360 250" style="width: 100%; max-height: 260px">
    <rect x="0" y="0" :width="2 * STEP + SIZE" :height="STEP + SIZE" fill="#eef3fd" />
    <g v-for="r in rows" :key="r">
      <rect v-for="c in cols" :key="c" :x="c * STEP" :y="r * STEP" :width="SIZE" :height="SIZE"
            fill="none" stroke="#4f7db3" stroke-width="1.6" stroke-dasharray="5 4" />
    </g>
    <rect :x="STEP" :y="STEP" :width="SIZE - STEP" :height="SIZE - STEP" fill="#4f7db3" opacity="0.25" />
    <text :x="STEP + 15" :y="STEP + SIZE + 46" text-anchor="middle" fill="#666" style="font-size: 13px">overlaps are blended at every step</text>
    <line :x1="STEP + 15" :y1="STEP + SIZE + 32" :x2="STEP + 15" :y2="STEP + 18" stroke="#888" stroke-width="1" />
  </svg>
</template>
