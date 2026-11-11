<!-- The story of the talk as five steps. `current` lights one up; `loop` draws the arrow back to the start. -->
<script setup lang="ts">
import { PARTS } from '../parts'
defineProps<{ current?: number; loop?: boolean }>()
</script>

<template>
  <div class="arc">
    <template v-for="(part, i) in PARTS" :key="i">
      <div class="step" :class="{ now: current === i + 1, done: current && i + 1 < current }">
        <div class="n">{{ i + 1 }}</div>
        <div class="t">{{ part.step }}</div>
      </div>
      <div v-if="i < PARTS.length - 1" class="link" :class="{ done: current && i + 1 < current }" />
    </template>
    <svg v-if="loop" class="back" viewBox="0 0 868 60" preserveAspectRatio="none">
      <path d="M 800 4 C 800 52, 760 52, 434 52 S 68 52, 68 12" fill="none" stroke="#0f8a7a" stroke-width="2.2" stroke-dasharray="5 5" />
      <path d="M 61 22 L 68 8 L 75 22" fill="none" stroke="#0f8a7a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  </div>
</template>

<style scoped>
.arc { display: flex; align-items: flex-start; width: 868px; position: relative; padding-bottom: 64px; }
.step { width: 136px; flex: none; }
.n {
  width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 15px; font-weight: 650; background: var(--wash); color: var(--muted); margin-bottom: 10px;
}
.t { font-size: 15.5px; line-height: 1.3; color: var(--muted); font-weight: 500; padding-right: 14px; }
.link { flex: 1; height: 2px; background: var(--faint); margin-top: 16px; margin-left: -96px; margin-right: 6px; }
.step.done .n { background: var(--ink); color: #fff; }
.step.done .t { color: var(--ink); }
.link.done { background: var(--ink); }
.step.now .n { background: var(--accent); color: #fff; box-shadow: 0 0 0 5px #0f8a7a22; }
.step.now .t { color: var(--ink); font-weight: 650; }
.back { position: absolute; left: 0; bottom: 0; width: 868px; height: 60px; }
</style>
