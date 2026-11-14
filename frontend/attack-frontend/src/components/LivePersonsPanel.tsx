import { useEffect, useState } from 'react'
import { Activity, Cpu } from 'lucide-react'
import { ApiError, apiFetch } from '../api/client'
import { ACTION_NAMES, INDEX_META, LEVEL_META } from '../constants/risk'
import type { LiveSnapshot } from '../types/intent'
import IndexBars from './IndexBars'
import RiskBadge from './RiskBadge'
import Sparkline from './Sparkline'

type LivePersonsPanelProps = {
  taskId: string
}

/** 即時人物風險：每秒輪詢 /api/intent/live/{taskId} */
export default function LivePersonsPanel({ taskId }: LivePersonsPanelProps) {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<number | null>(null)

  useEffect(() => {
    setSnap(null)
    setError('')
    if (!taskId) return
    let alive = true
    const load = async () => {
      try {
        const data = await apiFetch<LiveSnapshot>(`/api/intent/live/${taskId}`)
        if (alive) {
          setSnap(data)
          setError('')
        }
      } catch (err) {
        if (alive) setError(err instanceof ApiError && err.status === 404 ? '分析引擎啟動中…' : '無法取得即時分析')
      }
    }
    void load()
    const timer = window.setInterval(load, 1000)
    return () => {
      alive = false
      window.clearInterval(timer)
    }
  }, [taskId])

  if (!taskId) {
    return <p className="py-6 text-center text-[11px] text-slate-500">開始監測後顯示畫面中人物的即時風險</p>
  }
  if (!snap) {
    return <p className="py-6 text-center text-[11px] text-slate-500">{error || '載入中…'}</p>
  }

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-[10px] text-slate-500">
        <Cpu className="h-3 w-3" />
        <span>{snap.model_loaded ? 'ST-GCN + LSTM + AHP' : '規則 + AHP 基線（未載入模型）'}</span>
        <span className="ml-auto tabular-nums">
          {snap.persons.length} 人 ・ 告警 {snap.alerts_emitted}
        </span>
      </div>
      {snap.persons.length === 0 ? (
        <p className="py-4 text-center text-[11px] text-slate-500">畫面中目前沒有追蹤中的人物</p>
      ) : (
        snap.persons.map((p) => {
          const open = expanded === p.person_id
          return (
            <div
              key={p.person_id}
              className={`mb-2 rounded-lg border bg-white/60 p-2 ${LEVEL_META[p.level].border}`}
            >
              <button
                type="button"
                className="flex w-full items-center gap-2 text-left"
                onClick={() => setExpanded(open ? null : p.person_id)}
              >
                <RiskBadge level={p.level} />
                <span className="text-xs font-bold text-slate-900">P{p.person_id}</span>
                <span className="text-[10px] text-slate-500">{p.camera_id}</span>
                <span className="ml-auto text-[11px] font-bold tabular-nums text-slate-700">R {p.score.toFixed(2)}</span>
              </button>
              <div className="mt-1 flex items-center gap-2">
                <Sparkline points={p.history} color={LEVEL_META[p.level].hex} />
                <div className="flex min-w-0 flex-1 flex-wrap gap-1">
                  {p.top_indices.slice(0, 2).map((t) => (
                    <span key={t.key} className="rounded bg-slate-900/5 px-1 py-0.5 text-[10px] text-slate-700">
                      {t.key} {INDEX_META[t.key].name} {t.value.toFixed(2)}
                    </span>
                  ))}
                  {p.action && (
                    <span className="rounded bg-indigo-100 px-1 py-0.5 text-[10px] text-indigo-800">
                      <Activity className="mr-0.5 inline h-2.5 w-2.5" />
                      {ACTION_NAMES[p.action] ?? p.action}
                    </span>
                  )}
                </div>
              </div>
              {open && (
                <div className="mt-2 border-t border-slate-200 pt-2">
                  <IndexBars values={p.indices} predicted={p.predicted_next} compact />
                  {p.triggers.length > 0 && (
                    <ul className="mt-2 space-y-0.5 text-[10px] text-red-700">
                      {p.triggers.map((t) => (
                        <li key={t.rule}>▲ {t.description}（{t.duration}s）</li>
                      ))}
                    </ul>
                  )}
                  {p.camera_path.length > 1 && (
                    <p className="mt-1 text-[10px] text-slate-500">
                      跨鏡頭：{p.camera_path.map((c) => c.camera_id).join(' → ')}
                    </p>
                  )}
                  {!p.has_pose && <p className="mt-1 text-[10px] text-amber-700">此幀未取得骨架（人物過小或遮擋）</p>}
                </div>
              )}
            </div>
          )
        })
      )}
    </div>
  )
}
