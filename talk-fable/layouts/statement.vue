<!-- One sentence, alone. Used twice: the question the talk turns on, and the questions for the room. -->
<script setup lang="ts">
import { PARTS } from '../parts'
</script>

<template>
  <div class="slidev-layout">
    <div class="frame" style="justify-content: center">
      <div class="body" style="justify-content: center"><slot /></div>
      <div class="foot">
        <span v-if="$frontmatter.part"><b>{{ $frontmatter.part }}</b>&ensp;{{ PARTS[$frontmatter.part - 1].short }}</span>
        <span v-else />
        <span>{{ $page }}</span>
      </div>
    </div>
  </div>
</template>
