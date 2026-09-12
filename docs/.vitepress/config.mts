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
      { text: '第 3 章 GPU 入门', link: '/ch/part1/第3章_GPU入门' },
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
  { text: 'Part I', link: '/eng/part1/Chapter1_Introduction_to_LLM' },
  { text: 'Part II', link: '/eng/part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like' },
  { text: 'Part III', link: '/eng/part3/Chapter1_Attention-Backends-and-CUDA-Graph' },
  { text: 'Part IV', link: '/eng/part4/Chapter1_Deploying-SGLang-with-the-Cookbook' },
  { text: 'Community', link: '/eng/community/' },
]

// Every chapter has a page; untranslated ones are placeholders (see course-material/eng/WRITING_TEMPLATE.md).
const engSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Part 0 — Before you learn',
    items: [
      { text: 'Coding Ethics and Open-Source Spirit', link: '/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit' },
      { text: 'Deploy Your First SGLang Server', link: '/eng/part0/Part0-Deploy-Your-First-SGLang-Server' },
    ],
  },
  {
    text: 'Part I — Foundations',
    items: [
      { text: 'Chapter 1 Introduction to LLM', link: '/eng/part1/Chapter1_Introduction_to_LLM' },
      { text: 'Chapter 2 Introduction to Inference', link: '/eng/part1/Chapter2_Introduction to Inference.md' },
      { text: 'Chapter 3 Introduction to GPU', link: '/eng/part1/Chapter3_Introduction_to_GPU' },
      { text: 'Chapter 4 KV Cache', link: '/eng/part1/Chapter4_KV Cache The Core Data Structure of Inference.md' },
      { text: 'Chapter 5 Introduction to Benchmark', link: '/eng/part1/Chapter5_Introduction_to_Benchmark' },
    ],
  },
  {
    text: 'Part II — Build Your Own Mini SGL',
    items: [
      { text: 'Chapter 1 mini-sglang Overview', link: '/eng/part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like' },
      { text: 'Chapter 2 Inside SGLang', link: '/eng/part2/Chapter2_Inside-SGLang' },
      { text: 'Chapter 3 Forward Pass and Generation', link: '/eng/part2/Chapter3_Forward-Pass-and-Generation' },
      { text: 'Chapter 4 KV Cache Optimization', link: '/eng/part2/Chapter4_KV-Cache-Optimization' },
      { text: 'Chapter 5 HTTP and Concurrent Requests', link: '/eng/part2/Chapter5_HTTP-and-Concurrent-Requests' },
      { text: 'Chapter 6 Continuous Batching and the Scheduler', link: '/eng/part2/Chapter6_Continuous-Batching-and-the-Scheduler' },
      { text: 'Chapter 7 Paged KV Cache and Memory Management', link: '/eng/part2/Chapter7_Paged-KV-Cache-and-Memory-Management' },
      { text: 'Chapter 8 RadixAttention and Prefix Caching', link: '/eng/part2/Chapter8_RadixAttention-and-Prefix-Caching' },
      { text: 'Chapter 9 Multi-process and Tensor Parallelism', link: '/eng/part2/Chapter9_Multi-process-and-Tensor-Parallelism' },
      { text: 'Chapter 10 Speculative Decoding', link: '/eng/part2/Chapter10_Speculative-Decoding' },
    ],
  },
  {
    text: 'Part III — Advanced Inference Technique',
    items: [
      { text: 'Chapter 1 Attention Backends and CUDA Graph', link: '/eng/part3/Chapter1_Attention-Backends-and-CUDA-Graph' },
      { text: 'Chapter 2 Quantization and Low-Precision Inference', link: '/eng/part3/Chapter2_Quantization-and-Low-Precision-Inference' },
      { text: 'Chapter 3 Hierarchical Caching', link: '/eng/part3/Chapter3_Hierarchical-Caching' },
      { text: 'Chapter 4 Scaling Out', link: '/eng/part3/Chapter4_Scaling-Out' },
      { text: 'Chapter 5 Prefill-Decode Disaggregation', link: '/eng/part3/Chapter5_Prefill-Decode-Disaggregation' },
    ],
  },
  {
    text: 'Part IV — How to Make Contribution to SGLang',
    items: [
      { text: 'Chapter 1 Deploying SGLang with the Cookbook', link: '/eng/part4/Chapter1_Deploying-SGLang-with-the-Cookbook' },
      { text: 'Chapter 2 Profiling and Trace Analysis', link: '/eng/part4/Chapter2_Profiling-and-Trace-Analysis' },
      { text: 'Chapter 3 SGLang PR Workflow', link: '/eng/part4/Chapter3_SGLang-PR-Workflow' },
    ],
  },
  {
    text: 'Community',
    items: [
      { text: 'About', link: '/eng/community/' },
      { text: 'PR requirements', link: '/eng/community/PR_requirement' },
    ],
  },
]

// ---------------------------------------------------------------------------
// Source layout -> site URL
// ---------------------------------------------------------------------------
//
//   course-material/<lang>/...  ->  /<lang>/...
//   community/<lang>/...        ->  /<lang>/community/...
//
// Locales are keyed by the first URL segment, and public URLs stay exactly
// /ch/... and /eng/... regardless of where the sources live.
const SOURCE_TO_URL: [RegExp, string][] = [
  [/^course-material\/(ch|eng)\//, '$1/'],
  [/^community\/(ch|eng)\//, '$1/community/'],
]

function toSiteUrl(id: string): string {
  for (const [re, to] of SOURCE_TO_URL) {
    if (re.test(id)) return id.replace(re, to)
  }
  return id
}

// ---------------------------------------------------------------------------
// Site
// ---------------------------------------------------------------------------

export default defineConfig({
  title: 'zero-to-sglang',
  base,
  // Content lives in <repo>/course-material and <repo>/community; this folder only holds the site config.
  srcDir: '..',
  srcExclude: ['*.md', '**/WRITING_TEMPLATE.md', '.github/**', 'docs/**'],
  rewrites: toSiteUrl,
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
  // while the site root lands on the English edition.
  buildEnd(siteConfig) {
    for (const page of siteConfig.pages) {
      const url = siteConfig.rewrites.map[page] ?? page
      if (!url.startsWith('ch/')) continue
      const rest = url.slice('ch/'.length).replace(/\.md$/, '.html')
      const isHome = rest === 'index.html'
      const target = isHome ? `${base}eng/` : `${base}ch/${rest}`
      const href = encodeURI(target)
      const out = join(siteConfig.outDir, rest)
      mkdirSync(dirname(out), { recursive: true })
      writeFileSync(
        out,
        `<!DOCTYPE html><html lang="${isHome ? 'en-US' : 'zh-CN'}"><head><meta charset="utf-8">` +
          `<title>Redirecting…</title>` +
          `<meta http-equiv="refresh" content="0; url=${href}">` +
          `<link rel="canonical" href="${href}">` +
          `<script>location.replace(${JSON.stringify(href)})</script>` +
          `</head><body><a href="${href}">Redirecting…</a></body></html>\n`,
      )
    }
  },
})
