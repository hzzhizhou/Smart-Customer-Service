<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="brand">
        <span class="brand-logo">智</span>
        <span class="brand-title">智能售后客服</span>
      </div>

      <el-tabs v-model="tab" class="auth-tabs" stretch>
        <!-- 登录 -->
        <el-tab-pane label="登录" name="login">
          <el-form :model="loginForm" label-position="top" @submit.prevent>
            <el-form-item label="用户名">
              <el-input v-model="loginForm.username" placeholder="请输入用户名" size="large" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input
                v-model="loginForm.password"
                type="password"
                show-password
                placeholder="请输入密码"
                size="large"
                @keyup.enter="doLogin"
              />
            </el-form-item>
            <el-button
              type="primary"
              class="submit-btn"
              size="large"
              :loading="loading"
              @click="doLogin"
            >
              登录
            </el-button>
            <p class="tip">还没有账号？<el-link type="primary" @click="tab = 'register'">立即注册</el-link></p>
          </el-form>
        </el-tab-pane>

        <!-- 注册 -->
        <el-tab-pane label="注册" name="register">
          <el-form :model="regForm" label-position="top" @submit.prevent>
            <el-form-item label="用户名">
              <el-input v-model="regForm.username" placeholder="3-20 个字符" size="large" />
            </el-form-item>
            <el-form-item label="昵称（可选）">
              <el-input v-model="regForm.nickname" placeholder="不填则与用户名相同" size="large" />
            </el-form-item>
            <el-form-item label="密码">
              <el-input
                v-model="regForm.password"
                type="password"
                show-password
                placeholder="至少 6 位"
                size="large"
              />
            </el-form-item>
            <el-form-item label="确认密码">
              <el-input
                v-model="regForm.confirm"
                type="password"
                show-password
                placeholder="请再次输入密码"
                size="large"
                @keyup.enter="doRegister"
              />
            </el-form-item>
            <el-button
              type="primary"
              class="submit-btn"
              size="large"
              :loading="loading"
              @click="doRegister"
            >
              注册并登录
            </el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>

      <p class="tip muted back">登录后即可与智能客服对话</p>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const tab = ref(route.query.tab === 'register' ? 'register' : 'login')
const loading = ref(false)
const loginForm = reactive({ username: '', password: '' })
const regForm = reactive({ username: '', nickname: '', password: '', confirm: '' })

function afterAuth() {
  const redirect = route.query.redirect
  // 带了来源页且自己有权限，就回到来源页；普通用户无权进后台，直接送回用户端首页
  if (redirect && (userStore.isAdmin || !String(redirect).startsWith('/admin'))) {
    router.replace(redirect)
    return
  }
  router.replace(userStore.isAdmin ? '/admin/tickets' : '/')
}

async function doLogin() {
  if (!loginForm.username || !loginForm.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    await userStore.login(loginForm.username.trim(), loginForm.password)
    ElMessage.success(`欢迎回来，${userStore.displayName}`)
    afterAuth()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '登录失败，请稍后重试')
  } finally {
    loading.value = false
  }
}

async function doRegister() {
  const username = regForm.username.trim()
  if (username.length < 3) {
    ElMessage.warning('用户名至少 3 个字符')
    return
  }
  if (regForm.password.length < 6) {
    ElMessage.warning('密码至少 6 位')
    return
  }
  if (regForm.password !== regForm.confirm) {
    ElMessage.warning('两次输入的密码不一致')
    return
  }
  loading.value = true
  try {
    await userStore.register(username, regForm.password, regForm.nickname.trim())
    ElMessage.success('注册成功，已自动登录')
    afterAuth()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '注册失败，请稍后重试')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-page {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(160deg, #eef2fb 0%, #dfe7f7 100%);
}
.auth-card {
  width: 400px;
  padding: 32px 34px 22px;
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 10px 34px rgba(31, 45, 61, 0.12);
}
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.brand-logo {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, #409eff, #6a5cff);
  color: #fff; display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 700;
}
.brand-title { font-size: 16px; font-weight: 600; color: #1f2d3d; }
.submit-btn { width: 100%; margin-top: 4px; }
.tip { text-align: center; margin: 14px 0 0; font-size: 13px; color: #606266; }
.tip.muted { color: #a8abb2; font-size: 12px; }
.back { margin-top: 18px; }
</style>