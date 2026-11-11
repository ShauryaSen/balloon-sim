<!-- The title slide: the question, who is asking, and real wind underneath. -->
<template>
  <div class="slidev-layout">
    <div style="position: absolute; inset: 0; display: flex; flex-direction: column">
      <div style="flex: 1; padding: 64px 56px 0; display: flex; flex-direction: column; justify-content: center">
        <div style="font-size: 46px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; max-width: 780px">
          <slot name="title">{{ $frontmatter.title }}</slot>
        </div>
        <div style="font-size: 21px; color: var(--muted); margin-top: 16px; max-width: 700px">{{ $frontmatter.subtitle }}</div>
        <div style="font-size: 17px; margin-top: 30px"><b style="font-weight: 600">{{ $frontmatter.authors }}</b></div>
        <div style="font-size: 15px; color: var(--muted); margin-top: 4px">{{ $frontmatter.venue }}</div>
      </div>
      <img src="/fig/cover.png" style="width: 100%; height: 188px; object-fit: cover" />
      <div style="position: absolute; right: 16px; bottom: 10px; font-size: 11px; color: #ffffffcc">ERA5 wind at 19 km, 16 July 2021</div>
    </div>
  </div>
</template>
