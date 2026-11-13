<!--
  The learned controller as one picture: what the balloon sees (the forecast at each altitude, the goal),
  the shared policy network, and what it does (climb or sink). Clicks: 1 the network, 2 the action.
-->
<script setup lang="ts">
// forecast wind at each altitude, from the top down (+ east, - west, size is speed)
const WINDS = [0.9, 0.45, -0.3, -0.75, 0.2, 0.65]
const LANE = (i: number) => 46 + i * 40
const BALLOON = 3 // which lane the balloon is in
const LAYERS = [[60, 100, 140, 180, 220, 260], [80, 125, 170, 215, 250], [130, 190]].map((ys, i) => ys.map(y => ({ x: 430 + i * 85, y })))
const edges = LAYERS.slice(0, -1).flatMap((layer, i) => layer.flatMap(a => LAYERS[i + 1].map(b => ({ a, b }))))
</script>

<template>
  <svg viewBox="0 0 880 330" style="width: 100%">
    <!-- what it sees -->
    <text x="150" y="18" text-anchor="middle" fill="#1c2433" style="font-size: 19px; font-weight: 650">Sees</text>
    <g v-for="(w, i) in WINDS" :key="i">
      <rect x="20" :y="LANE(i) - 16" width="260" height="32" rx="7" :fill="i === BALLOON ? '#eef3fd' : '#f4f5f7'" />
      <g :transform="`translate(150, ${LANE(i)})`">
        <line :x1="-60 * w" y1="0" :x2="60 * w" y2="0" stroke="#0f8a7a" stroke-width="3" stroke-linecap="round" />
        <path :d="`M ${60 * w - 9 * Math.sign(w)} -6 L ${60 * w} 0 L ${60 * w - 9 * Math.sign(w)} 6`" fill="none" stroke="#0f8a7a" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
      </g>
    </g>
    <g :transform="`translate(58, ${LANE(BALLOON)})`">
      <line x1="0" y1="2" x2="0" y2="14" stroke="#1c2433" stroke-width="1.6" />
      <circle r="11" fill="#2f6fde" />
    </g>
    <text x="150" y="300" text-anchor="middle" fill="#1c2433" style="font-size: 16px">forecast at every altitude, the goal,</text>
    <text x="150" y="320" text-anchor="middle" fill="#1c2433" style="font-size: 16px">its own wind reading, the fleet</text>

    <path d="M 300 166 H 382" stroke="#8a93a3" stroke-width="2.5" />
    <path d="M 374 159 L 384 166 L 374 173" fill="none" stroke="#8a93a3" stroke-width="2.5" stroke-linejoin="round" />

    <!-- the policy -->
    <g v-click="1">
      <text x="515" y="18" text-anchor="middle" fill="#1c2433" style="font-size: 19px; font-weight: 650">PPO policy</text>
      <line v-for="(e, k) in edges" :key="k" :x1="e.a.x" :y1="e.a.y" :x2="e.b.x" :y2="e.b.y" stroke="#c9ced6" stroke-width="1.1" />
      <g v-for="(layer, i) in LAYERS" :key="i">
        <circle v-for="(n, j) in layer" :key="j" :cx="n.x" :cy="n.y" r="10" fill="#1c2433" />
      </g>
      <text x="515" y="300" text-anchor="middle" fill="#1c2433" style="font-size: 16px">one network, shared by</text>
      <text x="515" y="320" text-anchor="middle" fill="#1c2433" style="font-size: 16px">every balloon in a fleet</text>
    </g>

    <!-- what it does -->
    <g v-click="2">
      <path d="M 640 166 H 702" stroke="#8a93a3" stroke-width="2.5" />
      <path d="M 694 159 L 704 166 L 694 173" fill="none" stroke="#8a93a3" stroke-width="2.5" stroke-linejoin="round" />
      <text x="790" y="18" text-anchor="middle" fill="#1c2433" style="font-size: 19px; font-weight: 650">Does</text>
      <path d="M 790 70 L 830 120 H 750 Z" fill="#2f6fde" />
      <text x="790" y="148" text-anchor="middle" fill="#1c2433" style="font-size: 18px; font-weight: 600">climb</text>
      <text x="790" y="200" text-anchor="middle" fill="#1c2433" style="font-size: 18px; font-weight: 600">sink</text>
      <path d="M 790 262 L 830 212 H 750 Z" fill="#2f6fde" />
      <text x="790" y="300" text-anchor="middle" fill="#1c2433" style="font-size: 16px">every 3 hours</text>
    </g>
  </svg>
</template>
