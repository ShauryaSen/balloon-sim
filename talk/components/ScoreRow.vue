<!--
  One flight's (or one average's) score as a bar: the fixed plan at the left end, hindsight at the right,
  MPC where it lands between them, and the share of the gap it closed.

    <ScoreRow title="Example flight" :fixed="2373" :mpc="644" :hindsight="503" />
-->
<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ title: string; fixed: number; mpc: number; hindsight: number }>()
const X0 = 250, X1 = 790
const share = computed(() => (props.fixed - props.mpc) / (props.fixed - props.hindsight))
const x = computed(() => X0 + share.value * (X1 - X0))
const km = (v: number) => `${v.toLocaleString()} km`
</script>

<template>
  <svg viewBox="0 0 860 120" style="width: 100%">
    <text x="0" y="56" fill="#1c2433" style="font-size: 20px; font-weight: 650">{{ title }}</text>
    <text x="0" y="96" fill="#2f6fde" style="font-size: 40px; font-weight: 700; letter-spacing: -1px">{{ Math.round(share * 100) }}%</text>

    <rect :x="X0" y="66" :width="X1 - X0" height="14" rx="7" fill="#e6e8ec" />
    <rect :x="X0" y="66" :width="x - X0" height="14" rx="7" fill="#2f6fde" opacity="0.28" />

    <circle :cx="X0" cy="73" r="12" fill="#e2852e" />
    <text :x="X0" y="30" fill="#e2852e" text-anchor="middle" style="font-size: 17px; font-weight: 650">fixed plan</text>
    <text :x="X0" y="50" fill="#1c2433" text-anchor="middle" style="font-size: 15px">{{ km(fixed) }}</text>

    <circle :cx="X1" cy="73" r="12" fill="#1c2433" />
    <text :x="X1" y="30" fill="#1c2433" text-anchor="middle" style="font-size: 17px; font-weight: 650">hindsight</text>
    <text :x="X1" y="50" fill="#1c2433" text-anchor="middle" style="font-size: 15px">{{ km(hindsight) }}</text>

    <circle :cx="x" cy="73" r="12" fill="#2f6fde" />
    <text :x="x" y="108" fill="#2f6fde" text-anchor="middle" style="font-size: 17px; font-weight: 650">MPC <tspan fill="#1c2433" style="font-weight: 400; font-size: 15px">{{ km(mpc) }}</tspan></text>
  </svg>
</template>
