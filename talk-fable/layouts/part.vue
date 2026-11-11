<!-- The slide that opens each part: where we are in the story, and the question this part answers. -->
<script setup lang="ts">
import { PARTS } from '../parts'
</script>

<template>
  <div class="slidev-layout">
    <div class="frame" style="justify-content: center">
      <div class="muted" style="font-size: 15px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase">
        Part {{ $frontmatter.part }}<span v-if="$frontmatter.who"> · {{ $frontmatter.who }}</span>
      </div>
      <div style="font-size: 40px; font-weight: 650; letter-spacing: -0.025em; line-height: 1.15; margin: 10px 0 44px; max-width: 760px">
        {{ PARTS[$frontmatter.part - 1].question }}
      </div>
      <Arc :current="$frontmatter.part" />
      <slot />
    </div>
  </div>
</template>
