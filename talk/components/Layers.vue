<!--
  A figure saved as a stack of pictures (figures/style.py, save_layers), revealed one click at a time.

    <Layers name="bounds" :n="5" />              every layer on its own click
    <Layers name="bounds" :n="5" :shown="2" />   the first two layers are there from the start
    <Layers name="realism" :n="1" />             only the first layer, ever
    <Layers name="a" :n="5" :shown="1" at />     clicks at fixed indices, so two figures on one slide reveal together
-->
<script setup lang="ts">
withDefaults(defineProps<{ name: string; n: number; shown?: number; width?: string; at?: boolean }>(), { shown: 0, width: '100%', at: false })
</script>

<template>
  <div class="layers" :style="{ width }">
    <img :src="`/fig/${name}-0.png`" class="base" />
    <template v-for="i in n" :key="i">
      <img v-if="i <= shown" :src="`/fig/${name}-${i}.png`" class="over" />
      <img v-else-if="at" v-click="i - shown" :src="`/fig/${name}-${i}.png`" class="over" />
      <img v-else v-click :src="`/fig/${name}-${i}.png`" class="over" />
    </template>
  </div>
</template>

<style scoped>
.layers { position: relative; }
.base { width: 100%; }
.over { position: absolute; inset: 0; width: 100%; height: 100%; }
</style>
