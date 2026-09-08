import { defineConfig, type DefaultTheme } from 'vitepress'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

const base = '/zero-to-sglang/'
const repo = 'https://github.com/datawhalechina/zero-to-sglang'

// ---------------------------------------------------------------------------
// 简体中文（/ch/）
// ---------------------------------------------------------------------------

const chNav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/ch/' },
  { text: 'Part 0', link: '/ch/part0/Part0-编码伦理与开源精神' },
  { text: 'Part I', link: '/ch/part1/第1章_LLM入门' },
  { text: 'Part II', link: '/ch/part2/第1章_mini-sglang：推理引擎长什么样' },
  { text: 'Part III', link: '/ch/part3/第1章_AttentionBackends与CUDAGraph' },
  { text: 'Part IV', link: '/ch/part4/第1章_用Cookbook部署SGLang' },
  { text: '社区贡献', link: '/ch/community/' },
]

const chSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Part 0 — Before you learn',
    items: [
      { text: '编码伦理与开源精神', link: '/ch/part0/Part0-编码伦理与开源精神' },
      { text: '部署你的第一个 SGLang 服务', link: '/ch/part0/Part0-部署你的第一个SGLang服务' },
    ],
  },
  {
    text: 'Part I — Foundations',
    items: [
      { text: '第 1 章 LLM 入门', link: '/ch/part1/第1章_LLM入门' },
      { text: '第 2 章 推理入门', link: '/ch/part1/第2章_推理入门' },
      { text: '第 3 章 GPU 入门', link: '/ch/part1/第三章_IntroductionToGPU' },
      { text: '第 4 章 KV Cache', link: '/ch/part1/第4章_推理的核心数据结构入门' },
      { text: '第 5 章 Benchmark 入门', link: '/ch/part1/第5章_Benchmark入门' },
    ],
  },
  {
    text: 'Part II — Build Your Own Mini SGL',
    items: [
      { text: '第 1 章 mini-sglang 概览', link: '/ch/part2/第1章_mini-sglang：推理引擎长什么样' },
      { text: '第 2 章 Inside SGLang', link: '/ch/part2/第2章_InsideSGLang' },
      { text: '第 3 章 前向与生成', link: '/ch/part2/第3章_前向与生成' },
      { text: '第 4 章 KV Cache 优化', link: '/ch/part2/第4章_KVCache优化' },
      { text: '第 5 章 HTTP 服务与并发', link: '/ch/part2/第5章_HTTP服务与并发' },
      { text: '第 6 章 Continuous Batching 与调度', link: '/ch/part2/第6章_ContinuousBatching与调度' },
      { text: '第 7 章 Paged KV Cache 与显存管理', link: '/ch/part2/第7章_PagedKVCache与显存管理' },
      { text: '第 8 章 RadixAttention 与前缀缓存', link: '/ch/part2/第8章_RadixAttention与前缀缓存' },
      { text: '第 9 章 多进程与张量并行', link: '/ch/part2/第9章_多进程与张量并行' },
      { text: '第 10 章 投机解码', link: '/ch/part2/第10章_投机解码' },
    ],
  },
  {
    text: 'Part III — Advanced Inference Technique',
    items: [
      { text: '第 1 章 Attention Backends 与 CUDA Graph', link: '/ch/part3/第1章_AttentionBackends与CUDAGraph' },
      { text: '第 2 章 量化与低精度推理', link: '/ch/part3/第2章_量化与低精度推理' },
      { text: '第 3 章 分层缓存', link: '/ch/part3/第3章_分层缓存' },
      { text: '第 4 章 横向扩展', link: '/ch/part3/第4章_横向扩展' },
      { text: '第 5 章 Prefill-Decode 分离', link: '/ch/part3/第5章_PrefillDecode分离' },
    ],
  },
  {
    text: 'Part IV — How to Make Contribution to SGLang',
    items: [
      { text: '第 1 章 用 Cookbook 部署 SGLang', link: '/ch/part4/第1章_用Cookbook部署SGLang' },
      { text: '第 2 章 Profiling 与 Trace 分析', link: '/ch/part4/第2章_Profiling与Trace分析' },
      { text: '第 3 章 SGLang PR 工作流', link: '/ch/part4/第3章_SGLangPR工作流' },
    ],
  },
  {
    text: 'Community — 社区贡献',
    items: [
      { text: '专区说明', link: '/ch/community/' },
      { text: 'PR 要求', link: '/ch/community/PR_requirement' },
    ],
  },
]

// ---------------------------------------------------------------------------
// English（/eng/）
// ---------------------------------------------------------------------------

const engNav: DefaultTheme.NavItem[] = [
  { text: 'Home', link: '/eng/' },
  { text: 'Part 0', link: '/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit' },
]

// Sections are added here as their English chapters land.
const engSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Part 0 — Before you learn',
    items: [
      { text: 'Coding Ethics and Open-Source Spirit', link: '/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit' },
      { text: 'Deploy Your First SGLang Server', link: '/eng/part0/Part0-Deploy-Your-First-SGLang-Server' },
    ],
  },
]

// ---------------------------------------------------------------------------
// Site
// ---------------------------------------------------------------------------

export default defineConfig({
  title: 'zero-to-sglang',
  base,
  // Content lives in <repo>/ch and <repo>/eng; this folder only holds the site config.
  srcDir: '..',
  srcExclude: ['*.md', '**/WRITING_TEMPLATE.md', '.github/**', 'docs/**'],
  ignoreDeadLinks: true,

  head: [
    ['meta', { name: 'theme-color', content: '#2563eb' }],
  ],

  markdown: {
    math: true,
  },

  locales: {
    ch: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/ch/',
      description: '《从零手搓SGLang》—— 从零构建 LLM 推理引擎',
      themeConfig: {
        nav: chNav,
        sidebar: chSidebar,
        outlineTitle: '本页目录',
        docFooter: { prev: '上一页', next: '下一页' },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        langMenuLabel: '切换语言',
      },
    },
    eng: {
      label: 'English',
      lang: 'en-US',
      link: '/eng/',
      description: 'Build an LLM inference engine from scratch, then read the real SGLang source',
      themeConfig: {
        nav: engNav,
        sidebar: engSidebar,
        outlineTitle: 'On this page',
      },
    },
  },

  themeConfig: {
    // File names differ between editions, so the language switcher goes to the
    // target edition's home page instead of a same-path page that may not exist.
    i18nRouting: false,

    search: {
      provider: 'local',
    },

    socialLinks: [
      { icon: 'github', link: repo },
    ],

    outline: {
      level: [2, 3],
    },
  },

  // The Chinese edition used to live at the site root (/part1/..., /community/...).
  // Emit a redirect stub at every old URL so links shared before the move keep working,
  // and make the site root land on the Chinese edition.
  buildEnd(siteConfig) {
    for (const page of siteConfig.pages) {
      if (!page.startsWith('ch/')) continue
      const rest = page.slice('ch/'.length).replace(/\.md$/, '.html')
      const target = rest === 'index.html' ? `${base}ch/` : `${base}ch/${rest}`
      const href = encodeURI(target)
      const out = join(siteConfig.outDir, rest)
      mkdirSync(dirname(out), { recursive: true })
      writeFileSync(
        out,
        `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">` +
          `<title>Redirecting…</title>` +
          `<meta http-equiv="refresh" content="0; url=${href}">` +
          `<link rel="canonical" href="${href}">` +
          `<script>location.replace(${JSON.stringify(href)})</script>` +
          `</head><body><a href="${href}">Redirecting…</a></body></html>\n`,
      )
    }
  },
})
