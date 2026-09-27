<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const MIN_SCALE = 0.1
const MAX_SCALE = 8

const isOpen = ref(false)
const imageSrc = ref('')
const imageAlt = ref('')
const image = ref<HTMLImageElement | null>(null)
const stage = ref<HTMLElement | null>(null)
const scale = ref(1)
const offset = ref({ x: 0, y: 0 })
const isDragging = ref(false)

let dragStart = { x: 0, y: 0 }
let dragOrigin = { x: 0, y: 0 }

const imageStyle = computed(() => ({
  transform: `translate(calc(-50% + ${offset.value.x}px), calc(-50% + ${offset.value.y}px)) scale(${scale.value})`,
}))

function clamp(value: number) {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, value))
}

function openImage(target: HTMLImageElement) {
  imageSrc.value = target.currentSrc || target.src
  imageAlt.value = target.alt || 'Image preview'
  isOpen.value = true
  scale.value = 1
  offset.value = { x: 0, y: 0 }

  nextTick(() => fitImage())
}

function closeImage() {
  isOpen.value = false
  isDragging.value = false
}

function fitImage() {
  if (!image.value || !stage.value || !image.value.naturalWidth || !image.value.naturalHeight) return

  const { clientWidth, clientHeight } = stage.value
  const horizontalFit = (clientWidth * 0.9) / image.value.naturalWidth
  const verticalFit = (clientHeight * 0.84) / image.value.naturalHeight

  scale.value = Math.min(1, horizontalFit, verticalFit)
  offset.value = { x: 0, y: 0 }
}

function setScale(nextScale: number, clientX?: number, clientY?: number) {
  if (!stage.value) return

  const previousScale = scale.value
  const next = clamp(nextScale)
  const rect = stage.value.getBoundingClientRect()
  const cursorX = (clientX ?? rect.left + rect.width / 2) - rect.left - rect.width / 2
  const cursorY = (clientY ?? rect.top + rect.height / 2) - rect.top - rect.height / 2
  const localX = (cursorX - offset.value.x) / previousScale
  const localY = (cursorY - offset.value.y) / previousScale

  scale.value = next
  offset.value = {
    x: cursorX - localX * next,
    y: cursorY - localY * next,
  }
}

function handleWheel(event: WheelEvent) {
  event.preventDefault()
  setScale(scale.value * Math.exp(-event.deltaY * 0.001), event.clientX, event.clientY)
}

function startDragging(event: PointerEvent) {
  if (event.button !== 0) return
  isDragging.value = true
  dragStart = { x: event.clientX, y: event.clientY }
  dragOrigin = { ...offset.value }
  stage.value?.setPointerCapture(event.pointerId)
}

function dragImage(event: PointerEvent) {
  if (!isDragging.value) return
  offset.value = {
    x: dragOrigin.x + event.clientX - dragStart.x,
    y: dragOrigin.y + event.clientY - dragStart.y,
  }
}

function stopDragging(event?: PointerEvent) {
  isDragging.value = false
  if (event && stage.value?.hasPointerCapture(event.pointerId)) {
    stage.value.releasePointerCapture(event.pointerId)
  }
}

function handleDocumentClick(event: MouseEvent) {
  const target = event.target
  if (!(target instanceof HTMLImageElement) || !target.closest('.vp-doc')) return
  event.preventDefault()
  openImage(target)
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && isOpen.value) closeImage()
}

onMounted(() => {
  document.addEventListener('click', handleDocumentClick)
  document.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleDocumentClick)
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="isOpen"
      class="image-lightbox"
      role="dialog"
      aria-modal="true"
      :aria-label="imageAlt"
      @click.self="closeImage"
    >
      <div
        ref="stage"
        class="image-lightbox__stage"
        :class="{ 'is-dragging': isDragging }"
        @wheel="handleWheel"
        @pointerdown="startDragging"
        @pointermove="dragImage"
        @pointerup="stopDragging"
        @pointercancel="stopDragging"
        @dblclick="fitImage"
      >
        <img
          ref="image"
          class="image-lightbox__image"
          :src="imageSrc"
          :alt="imageAlt"
          :style="imageStyle"
          draggable="false"
          @load="fitImage"
        >
      </div>

      <div class="image-lightbox__toolbar" aria-label="Image controls">
        <button type="button" aria-label="Zoom out" title="Zoom out" @click="setScale(scale - 0.2)">
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <path d="M10.5 24L38.5 24" />
          </svg>
        </button>
        <span>{{ Math.round(scale * 100) }}%</span>
        <button type="button" aria-label="Zoom in" title="Zoom in" @click="setScale(scale + 0.2)">
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <path d="M24.0605 10L24.0239 38" />
            <path d="M10 24L38 24" />
          </svg>
        </button>
        <button class="image-lightbox__reset" type="button" aria-label="Reset image" title="Reset image" @click="fitImage">
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <path d="M11.2721 36.7279C14.5294 39.9853 19.0294 42 24 42C33.9411 42 42 33.9411 42 24C42 14.0589 33.9411 6 24 6C19.0294 6 14.5294 8.01472 11.2721 11.2721C9.61407 12.9301 6 17 6 17" />
            <path d="M6 9V17H14" />
          </svg>
        </button>
      </div>

      <button class="image-lightbox__close" type="button" aria-label="Close image preview" @click="closeImage">
        <svg viewBox="0 0 48 48" aria-hidden="true">
          <path d="M8 8L40 40" />
          <path d="M8 40L40 8" />
        </svg>
      </button>
    </div>
  </Teleport>
</template>

<style scoped>
.image-lightbox {
  position: fixed;
  z-index: 100;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgb(15 23 42 / 82%);
  backdrop-filter: blur(3px);
}

.image-lightbox__stage {
  position: absolute;
  inset: 0;
  overflow: hidden;
  cursor: grab;
  touch-action: none;
}

.image-lightbox__stage.is-dragging {
  cursor: grabbing;
}

.image-lightbox__image {
  position: absolute;
  top: 50%;
  left: 50%;
  max-width: none;
  user-select: none;
  transform-origin: center;
  will-change: transform;
}

.image-lightbox__toolbar {
  position: fixed;
  bottom: 28px;
  left: 50%;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  color: #e2e8f0;
  background: rgb(15 23 42 / 88%);
  border: 1px solid rgb(148 163 184 / 30%);
  border-radius: 10px;
  transform: translateX(-50%);
}

.image-lightbox__toolbar button {
  min-width: 30px;
  height: 30px;
  padding: 0 8px;
  color: inherit;
  font: inherit;
  background: transparent;
  border: 0;
  border-radius: 6px;
  cursor: pointer;
}

.image-lightbox__toolbar button:hover,
.image-lightbox__toolbar button:focus-visible {
  background: rgb(148 163 184 / 25%);
  outline: none;
}

.image-lightbox__toolbar button svg {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.8;
  vertical-align: middle;
}

.image-lightbox__toolbar span {
  min-width: 50px;
  font-size: 13px;
  text-align: center;
}

.image-lightbox__reset {
  margin-left: 2px;
}

.image-lightbox__close {
  position: fixed;
  top: 18px;
  right: 22px;
  z-index: 1;
  width: 40px;
  height: 40px;
  display: grid;
  place-items: center;
  color: #e2e8f0;
  background: rgb(15 23 42 / 88%);
  border: 1px solid rgb(148 163 184 / 30%);
  border-radius: 50%;
  cursor: pointer;
}

.image-lightbox__close svg {
  width: 20px;
  height: 20px;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-width: 1.8;
}

.image-lightbox__close:hover,
.image-lightbox__close:focus-visible {
  background: #334155;
  outline: none;
}

@media (max-width: 640px) {
  .image-lightbox__toolbar {
    bottom: 36px;
  }
}
</style>
