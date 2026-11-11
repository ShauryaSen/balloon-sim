<!-- A grey box standing in for a figure we still have to make. Usage: <Todo>what goes here</Todo> -->
<template>
  <div class="todo"><slot /></div>
</template>

<style scoped>
.todo {
  border: 2px dashed #b0b0b0;
  color: #777;
  background: #fafafa;
  min-height: 260px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 1.5rem;
  font-size: 0.9em;
  margin-top: 1rem;
}
</style>
