<!--
  How a balloon steers: five altitudes, each with its own wind, and a balloon that can only go up or down.
  Adapted from the talk-fable draft, without the animation. Edit WINDS to change the winds.
-->
<script setup lang="ts">
// from the top altitude down: which way the wind blows (+ east, - west) and how hard (0 to 1)
const WINDS = [0.95, 0.35, -0.75, -0.25, 0.6]
const LANE = (i: number) => 52 + i * 62
const columns = [90, 230, 370, 510, 650]
const BALLOON = { x: 335, lane: 2 }
</script>

<template>
  <svg viewBox="-22 0 782 350" style="width: 100%; max-height: 330px">
    <defs>
      <marker id="red-head" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#c8372d" />
      </marker>
    </defs>
    <g v-for="(w, i) in WINDS" :key="i">
      <rect x="40" :y="LANE(i) - 27" width="700" height="54" rx="6" :fill="i === BALLOON.lane ? '#eef3fd' : '#f5f5f5'" />
      <g v-for="x in columns" :key="x" :transform="`translate(${x + 35}, ${LANE(i)})`" :opacity="0.35 + 0.5 * Math.abs(w)">
        <line :x1="-34 * w" y1="0" :x2="34 * w" y2="0" stroke="#888" stroke-width="2.4" stroke-linecap="round" />
        <path :d="`M ${34 * w - 9 * Math.sign(w)} -6 L ${34 * w} 0 L ${34 * w - 9 * Math.sign(w)} 6`" fill="none" stroke="#888" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />
      </g>
    </g>
    <text x="28" y="56" text-anchor="end" fill="#888" style="font-size: 13px">20 km</text>
    <text x="28" y="304" text-anchor="end" fill="#888" style="font-size: 13px">15 km</text>

    <!-- the only control: up or down -->
    <line :x1="BALLOON.x" :y1="LANE(BALLOON.lane) - 25" :x2="BALLOON.x" :y2="LANE(BALLOON.lane) - 78" stroke="#c8372d" stroke-width="3" marker-end="url(#red-head)" />
    <line :x1="BALLOON.x" :y1="LANE(BALLOON.lane) + 37" :x2="BALLOON.x" :y2="LANE(BALLOON.lane) + 90" stroke="#c8372d" stroke-width="3" marker-end="url(#red-head)" />

    <g :transform="`translate(${BALLOON.x}, ${LANE(BALLOON.lane)})`">
      <line x1="0" y1="14" x2="0" y2="27" stroke="#1a1a1a" stroke-width="1.6" />
      <rect x="-5" y="26" width="10" height="7" rx="1.5" fill="#1a1a1a" />
      <ellipse cx="0" cy="-3" rx="16" ry="18" fill="#2f6fde" />
      <ellipse cx="-5" cy="-9" rx="4" ry="6" fill="#ffffff" opacity="0.35" />
    </g>
  </svg>
</template>
