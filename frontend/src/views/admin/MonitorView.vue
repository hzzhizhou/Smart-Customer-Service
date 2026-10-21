<template>
  <div>
    <el-button type="primary" text @click="load">刷新</el-button>

    <el-card v-if="health" class="card">
      <template #header>系统健康</template>
      <el-descriptions :column="3" border>
        <el-descriptions-item label="状态">
          <el-tag :type="health.status === 'healthy' ? 'success' : 'danger'">
            {{ health.status }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item
          v-for="(v, k) in health.components"
          :key="k"
          :label="k"
        >
          <span :class="String(v).startsWith('ok') ? 'ok' : 'fail'">{{ v }}</span>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card class="card">
      <template #header>Prometheus 指标（关键项）</template>
      <el-descriptions :column="2" border>
        <el-descriptions-item label="http 请求总数">
          <b>{{ metrics.requests }}</b>
        </el-descriptions-item>
        <el-descriptions-item label="请求总耗时(s)">
          <b>{{ metrics.durationSum.toFixed(2) }}</b>
        </el-descriptions-item>
        <el-descriptions-item label="LLM 调用次数">
          <b>{{ metrics.llmRequests }}</b>
        </el-descriptions-item>
        <el-descriptions-item label="LLM token 合计">
          <b>{{ metrics.llmTokens }}</b>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import http from '@/api'

const health = ref(null)
const metrics = ref({ requests: 0, durationSum: 0, llmRequests: 0, llmTokens: 0 })

// 指标名必须与后端 utils/metrics.py 完全一致（统一 web_agent_ 前缀）；取值用浮点正则，
// prometheus_client 文本格式里 Counter 也输出 "1.0" 这样的浮点，整数正则会全部漏掉。
const PATTERNS = {
  requests: /^web_agent_http_requests_total\{.*\}\s+([\d.eE+-]+)$/,
  durationSum: /^web_agent_http_request_duration_seconds_sum\{.*\}\s+([\d.eE+-]+)$/,
  llmRequests: /^web_agent_llm_requests_total\{.*\}\s+([\d.eE+-]+)$/,
  llmTokens: /^web_agent_llm_tokens_total\{.*\}\s+([\d.eE+-]+)$/,
}

async function load() {
  try {
    const h = await http.get('/health')
    health.value = h.data
  } catch (e) {
    health.value = { status: 'unavailable: ' + (e?.response?.status || e.message), components: {} }
  }
  try {
    const m = await http.get('/metrics', { responseType: 'text' })
    const sum = { requests: 0, durationSum: 0, llmRequests: 0, llmTokens: 0 }
    for (const line of m.data.split('\n')) {
      for (const [key, re] of Object.entries(PATTERNS)) {
        const hit = line.match(re)
        if (hit) sum[key] += Number(hit[1])
      }
    }
    metrics.value = sum
  } catch (e) {
    /* metrics 读取失败忽略 */
  }
}

load()
</script>

<style scoped>
.card { margin-bottom: 16px; }
.ok { color: #67c23a; }
.fail { color: #f56c6c; }
</style>