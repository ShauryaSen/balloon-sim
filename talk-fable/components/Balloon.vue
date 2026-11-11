<!--
  How a balloon steers: five altitudes, each with its own wind, and a balloon that can only move between them.
  Each click is one decision. To change the story, edit WINDS (one wind per altitude) and STOPS (where the
  balloon is after each click).
-->
<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const { $clicks } = useSlideContext()

// from the top altitude down: which way the wind blows (+ east, - west) and how hard (0 to 1)
const WINDS = [0.95, 0.35, -0.75, -0.25, 0.6]
const LANE = (i: number) => 52 + i * 62
// after each click: [lane, x]
const STOPS: [number, number][] = [[2, 470], [2, 250], [0, 600], [3, 520]]

const step = computed(() => Math.min($clicks.value, STOPS.length - 1))
const at = computed(() => ({ x: STOPS[step.value][1], y: LANE(STOPS[step.value][0]) }))
const trail = computed(() => STOPS.slice(0, step.value + 1).map(([lane, x]) => `${x},${LANE(lane)}`).join(' '))
const columns = [90, 230, 370, 510, 650]
</script>

<template>
  <svg viewBox="-22 0 782 350" style="width: 100%; max-height: 350px">
    <g v-for="(w, i) in WINDS" :key="i">
      <rect x="40" :y="LANE(i) - 27" width="700" height="54" rx="8" :fill="i === STOPS[step][0] ? '#eef3fd' : '#f7f8fa'" style="transition: fill 400ms" />
      <g v-for="x in columns" :key="x" :transform="`translate(${x + 35}, ${LANE(i)})`" :opacity="0.35 + 0.5 * Math.abs(w)">
        <line :x1="-34 * w" y1="0" :x2="34 * w" y2="0" stroke="#8a93a3" stroke-width="2.4" stroke-linecap="round" />
        <path :d="`M ${34 * w - 9 * Math.sign(w)} -6 L ${34 * w} 0 L ${34 * w - 9 * Math.sign(w)} 6`" fill="none" stroke="#8a93a3" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />
      </g>
    </g>
    <text x="28" y="56" text-anchor="end" fill="#8a93a3" style="font-size: 13px">20 km</text>
    <text x="28" y="304" text-anchor="end" fill="#8a93a3" style="font-size: 13px">15 km</text>
    <polyline :points="trail" fill="none" stroke="#2f6fde" stroke-width="2.5" stroke-dasharray="2 7" stroke-linecap="round" />
    <g :style="{ transform: `translate(${at.x}px, ${at.y}px)`, transition: 'transform 1100ms cubic-bezier(.45,.05,.3,1)' }">
      <line x1="0" y1="14" x2="0" y2="27" stroke="#1c2433" stroke-width="1.6" />
      <rect x="-5" y="26" width="10" height="7" rx="1.5" fill="#1c2433" />
      <ellipse cx="0" cy="-3" rx="16" ry="18" fill="#2f6fde" />
      <ellipse cx="-5" cy="-9" rx="4" ry="6" fill="#ffffff" opacity="0.35" />
    </g>
  </svg>
</template>
