import type { EventStatus, IndexKey, RiskLevel } from '../types/intent'

export const LEVELS: RiskLevel[] = ['L0', 'L1', 'L2', 'L3', 'L4']

/** 研究計畫表 2：L0 藍／L1 綠／L2 黃／L3 橘／L4 紅 */
export const LEVEL_META: Record<
  RiskLevel,
  { color: string; reaction: string; badge: string; dot: string; border: string; soft: string; hex: string }
> = {
  L0: { color: '藍色', reaction: '持續監控', badge: 'bg-sky-500 text-white', dot: 'bg-sky-500', border: 'border-sky-300', soft: 'bg-sky-50', hex: '#0ea5e9' },
  L1: { color: '綠色', reaction: '記錄事件', badge: 'bg-emerald-500 text-white', dot: 'bg-emerald-500', border: 'border-emerald-300', soft: 'bg-emerald-50', hex: '#10b981' },
  L2: { color: '黃色', reaction: '標示', badge: 'bg-yellow-400 text-yellow-950', dot: 'bg-yellow-400', border: 'border-yellow-300', soft: 'bg-yellow-50', hex: '#eab308' },
  L3: { color: '橘色', reaction: '推播', badge: 'bg-orange-500 text-white', dot: 'bg-orange-500', border: 'border-orange-300', soft: 'bg-orange-50', hex: '#f97316' },
  L4: { color: '紅色', reaction: '立即警告', badge: 'bg-red-600 text-white', dot: 'bg-red-600', border: 'border-red-300', soft: 'bg-red-50', hex: '#dc2626' },
}

export const INDEX_KEYS: IndexKey[] = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']

/** 研究計畫表 1；threshold 為表 2 觸發規則使用的門檻 */
export const INDEX_META: Record<IndexKey, { name: string; threshold: number }> = {
  X1: { name: '武器準備', threshold: 0.7 },
  X2: { name: '拳擊姿態', threshold: 0.6 },
  X3: { name: '追逐衝刺', threshold: 0.5 },
  X4: { name: '徘徊', threshold: 0.5 },
  X5: { name: '尾隨', threshold: 0.5 },
  X6: { name: '距離侵略', threshold: 0.6 },
  X7: { name: '群聚對峙', threshold: 0.8 },
  X8: { name: '歷史風險', threshold: 0.6 },
  X9: { name: '空間脈絡', threshold: 0.5 },
}

export const STATUS_META: Record<EventStatus, { label: string; chip: string }> = {
  pending: { label: '未處理', chip: 'bg-slate-200 text-slate-700' },
  handled: { label: '已處理', chip: 'bg-emerald-100 text-emerald-800' },
  escalated: { label: '需升級', chip: 'bg-red-100 text-red-700' },
  dismissed: { label: '誤報', chip: 'bg-slate-100 text-slate-500' },
}

export const ACTION_NAMES: Record<string, string> = {
  normal: '正常',
  fight: '肢體攻擊',
  punch: '揮拳/掌摑',
  kick: '踢擊',
  push: '推撞',
  strike_object: '持物毆打',
  knife: '持刀',
  gun: '持槍',
  weapon: '持械攻擊',
  suspicious: '可疑',
  dangerous: '危險/暴力',
}

export function levelIndex(level: RiskLevel): number {
  return LEVELS.indexOf(level)
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '-'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('zh-TW', { hour12: false })
}

export function formatClock(iso: string | null | undefined): string {
  if (!iso) return '-'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString('zh-TW', { hour12: false })
}

export function topIndices(map: Record<string, number> | null | undefined, n = 2, min = 0.2) {
  if (!map) return []
  return INDEX_KEYS.map((k) => ({ key: k, value: Number(map[k] ?? 0) }))
    .filter((x) => x.value >= min)
    .sort((a, b) => b.value - a.value)
    .slice(0, n)
}
