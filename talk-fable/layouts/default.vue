<!-- Every content slide: the claim as the title, the visual in the middle, the part of the talk in the corner. -->
<script setup lang="ts">
import { PARTS } from '../parts'
</script>

<template>
  <div class="slidev-layout">
    <div class="frame">
      <h1 v-if="$frontmatter.claim">{{ $frontmatter.claim }}</h1>
      <p v-if="$frontmatter.lede" class="lede">{{ $frontmatter.lede }}</p>
      <div class="body"><slot /></div>
      <div class="foot">
        <span v-if="$frontmatter.part"><b>{{ $frontmatter.part }}</b>&ensp;{{ PARTS[$frontmatter.part - 1].short }}</span>
        <span v-else />
        <span>{{ $page }}</span>
      </div>
    </div>
  </div>
</template>
