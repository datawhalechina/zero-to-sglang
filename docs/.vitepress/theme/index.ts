import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import ImageLightbox from './ImageLightbox.vue'

import './style.css'

export default {
  ...DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, {
    'layout-bottom': () => h(ImageLightbox),
  }),
}
