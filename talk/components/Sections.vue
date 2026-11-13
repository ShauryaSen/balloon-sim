<!-- The talk's sections in a row. `current` is shown in black, the rest in grey. Usage: <Sections :current="2" /> -->
<script setup lang="ts">
import { SECTIONS } from '../sections'
defineProps<{ current?: number }>()
</script>

<template>
  <div class="sections">
    <div v-for="(name, i) in SECTIONS" :key="i" class="section" :class="{ now: current === i + 1 }">
      <div class="num">{{ i + 1 }}</div>
      <div class="name">{{ name }}</div>
    </div>
  </div>
</template>

<style scoped>
.sections { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-top: 2rem; }
.section { border-top: 3px solid #ddd; padding-top: 10px; color: #999; }
.num { font-size: 0.85rem; font-weight: 600; }
.name { font-size: 1rem; line-height: 1.3; }
.section.now { border-top-color: #1a1a1a; color: #1a1a1a; }
.section.now .name { font-weight: 600; }
</style>
