<!--
  The score: where MPC lands between the fixed plan and hindsight.
  The three numbers are the averages over 32 launches at a real forecast's error (3.2 m/s), printed by
  figures/results.py and drawn in the "bounds" figure. Change them here if the sweep changes.
-->
<script setup lang="ts">
const fixed = 1778, mpc = 1209, hindsight = 1043 // average distance from the goal, km
const share = (fixed - mpc) / (fixed - hindsight)
const X0 = 90, X1 = 770
const x = X0 + share * (X1 - X0)
</script>

<template>
  <svg viewBox="0 0 860 300" style="width: 100%">
    <text :x="X0" y="34" fill="#8a93a3" style="font-size: 15px">average distance from the goal over a flight</text>
    <text :x="X1" y="34" fill="#8a93a3" text-anchor="end" style="font-size: 15px">better →</text>
    <rect :x="X0" y="120" :width="X1 - X0" height="16" rx="8" fill="#e6e8ec" />

    <g>
      <circle :cx="X0" cy="128" r="13" fill="#e2852e" />
      <text :x="X0" y="88" fill="#e2852e" text-anchor="middle" style="font-size: 19px; font-weight: 650">fixed plan</text>
      <text :x="X0" y="108" fill="#8a93a3" text-anchor="middle" style="font-size: 14px">{{ fixed.toLocaleString() }} km</text>
      <circle :cx="X1" cy="128" r="13" fill="#1c2433" />
      <text :x="X1" y="88" fill="#1c2433" text-anchor="middle" style="font-size: 19px; font-weight: 650">hindsight</text>
      <text :x="X1" y="108" fill="#8a93a3" text-anchor="middle" style="font-size: 14px">{{ hindsight.toLocaleString() }} km</text>
    </g>

    <g v-click>
      <rect :x="X0" y="120" :width="x - X0" height="16" rx="8" fill="#2f6fde" opacity="0.28" />
      <circle :cx="x" cy="128" r="13" fill="#2f6fde" />
      <text :x="x" y="88" fill="#2f6fde" text-anchor="middle" style="font-size: 19px; font-weight: 650">MPC</text>
      <text :x="x" y="108" fill="#8a93a3" text-anchor="middle" style="font-size: 14px">{{ mpc.toLocaleString() }} km</text>
    </g>

    <g v-click>
      <path :d="`M ${X0} 162 v 12 H ${x} v -12`" fill="none" stroke="#2f6fde" stroke-width="2" />
      <text :x="(X0 + x) / 2" y="222" fill="#2f6fde" text-anchor="middle" style="font-size: 44px; font-weight: 700; letter-spacing: -1.5px">{{ Math.round(share * 100) }}%</text>
      <text :x="(X0 + x) / 2" y="250" fill="#1c2433" text-anchor="middle" style="font-size: 16px">of the gap closed, with a real forecast's error</text>
      <path :d="`M ${x} 162 v 12 H ${X1} v -12`" fill="none" stroke="#8a93a3" stroke-width="2" />
      <text :x="(x + X1) / 2" y="200" fill="#8a93a3" text-anchor="middle" style="font-size: 15px">what the wrong</text>
      <text :x="(x + X1) / 2" y="219" fill="#8a93a3" text-anchor="middle" style="font-size: 15px">forecast cost</text>
    </g>
  </svg>
</template>
