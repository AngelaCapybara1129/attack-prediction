import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { History, Search, X } from 'lucide-react'
import { apiFetch } from '../api/client'
import AlertCard from '../components/AlertCard'
import EventDetailDrawer from '../components/EventDetailDrawer'
import { INDEX_META, LEVELS, STATUS_META } from '../constants/risk'
import type { EventItem, EventListResponse, EventStatus, RiskLevel, TextSearchResponse } from '../types/intent'

const PAGE_SIZE = 30
const glassCard =
  'rounded-[16px] border border-white/50 bg-white/55 backdrop-blur-2xl shadow-[0_10px_30px_rgba(15,23,42,0.08)] ring-1 ring-white/60'

const EXAMPLES = ['上週穿黑衣揮拳', '深夜尾隨', '今天 L4', '持刀 cam0', '未處理 推擠']

/** 跨鏡頭軌跡追蹤與歷史證據回溯：事件列表 + 文字檢索 */
export default function EventsPage() {
  const [minLevel, setMinLevel] = useState<RiskLevel>('L1')
  const [status, setStatus] = useState<EventStatus | ''>('')
  const [page, setPage] = useState(0)
  const [list, setList] = useState<EventListResponse>({ total: 0, items: [] })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [query, setQuery] = useState('')
  const [search, setSearch] = useState<TextSearchResponse | null>(null)
  const [searching, setSearching] = useState(false)

  const [openId, setOpenId] = useState<number | null>(null)

  const loadList = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams({
        min_level: minLevel,
        limit: String(PAGE_SIZE),
        offset: String(page * PAGE_SIZE),
      })
      if (status) params.set('status', status)
      setList(await apiFetch<EventListResponse>(`/api/events?${params.toString()}`))
    } catch (err) {
      setError(err instanceof Error ? err.message : '讀取失敗')
    } finally {
      setLoading(false)
    }
  }, [minLevel, status, page])

  useEffect(() => {
    void loadList()
  }, [loadList])

  const runSearch = async (q: string) => {
    if (!q.trim()) {
      setSearch(null)
      return
    }
    setSearching(true)
    setError('')
    try {
      setSearch(
        await apiFetch<TextSearchResponse>('/api/search/text', {
          method: 'POST',
          body: JSON.stringify({ query: q.trim(), limit: 50 }),
        }),
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : '檢索失敗')
    } finally {
      setSearching(false)
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    void runSearch(query)
  }

  const onUpdated = (ev: EventItem) => {
    setList((prev) => ({ ...prev, items: prev.items.map((x) => (x.id === ev.id ? ev : x)) }))
    setSearch((prev) =>
      prev ? { ...prev, items: prev.items.map((it) => (it.event_id === ev.id ? { ...it, event: ev } : it)) } : prev,
    )
  }

  const totalPages = Math.max(1, Math.ceil(list.total / PAGE_SIZE))
  const parsed = search?.parsed

  return (
    <div className="mx-auto max-w-6xl p-3 md:p-5">
      <header className={`${glassCard} mb-4 p-4`}>
        <div className="mb-3 flex items-center gap-2">
          <History className="h-5 w-5 text-emerald-600" />
          <h1 className="text-base font-black text-slate-900">事件紀錄與證據回溯</h1>
        </div>
        <form onSubmit={onSubmit} className="flex flex-col gap-2 md:flex-row">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="以文字描述檢索，例如：上週穿黑衣揮拳、深夜尾隨、持刀 cam1"
              className="w-full rounded-lg border border-slate-200 bg-white/80 py-2 pl-9 pr-8 text-sm outline-none focus:border-emerald-400"
            />
            {query && (
              <button
                type="button"
                onClick={() => {
                  setQuery('')
                  setSearch(null)
                }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700"
                aria-label="清除"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={searching}
            className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-bold text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            {searching ? '檢索中…' : '檢索'}
          </button>
        </form>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuery(ex)
                void runSearch(ex)
              }}
              className="rounded-full bg-slate-900/5 px-2.5 py-1 text-[11px] text-slate-600 hover:bg-slate-900/10"
            >
              {ex}
            </button>
          ))}
        </div>
      </header>

      {error && <p className="mb-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}

      {search ? (
        <section className={`${glassCard} p-4`}>
          <div className="mb-3 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-600">
            <span className="font-bold text-slate-800">「{search.query}」</span>
            {parsed?.indices.map((k) => (
              <span key={k} className="rounded bg-orange-100 px-1.5 py-0.5 text-orange-800">
                {k} {INDEX_META[k].name}
              </span>
            ))}
            {parsed?.colors.map((c) => (
              <span key={c} className="rounded bg-slate-200 px-1.5 py-0.5">衣著：{c}</span>
            ))}
            {parsed?.levels.map((l) => (
              <span key={l} className="rounded bg-slate-200 px-1.5 py-0.5">{l}</span>
            ))}
            {parsed?.cameras.map((c) => (
              <span key={c} className="rounded bg-slate-200 px-1.5 py-0.5">{c}</span>
            ))}
            {parsed?.since && (
              <span className="rounded bg-slate-200 px-1.5 py-0.5">
                {new Date(parsed.since).toLocaleDateString('zh-TW')} 起
                {parsed.until ? ` 至 ${new Date(parsed.until).toLocaleDateString('zh-TW')}` : ''}
              </span>
            )}
            <span className="ml-auto">{search.items.length} 筆</span>
          </div>
          {search.items.length === 0 ? (
            <p className="py-8 text-center text-xs text-slate-500">沒有符合的事件</p>
          ) : (
            <div className="grid gap-2 md:grid-cols-2">
              {search.items.map((it) => (
                <AlertCard
                  key={it.event_id}
                  event={it.event}
                  onOpen={setOpenId}
                  extra={
                    it.reasons.length > 0 && (
                      <p className="mt-1 text-[10px] text-emerald-700">符合：{it.reasons.join('、')}</p>
                    )
                  }
                />
              ))}
            </div>
          )}
        </section>
      ) : (
        <section className={`${glassCard} p-4`}>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
            <label className="flex items-center gap-1 text-slate-600">
              最低等級
              <select
                value={minLevel}
                onChange={(e) => {
                  setPage(0)
                  setMinLevel(e.target.value as RiskLevel)
                }}
                className="rounded border border-slate-200 bg-white px-1.5 py-1"
              >
                {LEVELS.map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-slate-600">
              狀態
              <select
                value={status}
                onChange={(e) => {
                  setPage(0)
                  setStatus(e.target.value as EventStatus | '')
                }}
                className="rounded border border-slate-200 bg-white px-1.5 py-1"
              >
                <option value="">全部</option>
                {(Object.keys(STATUS_META) as EventStatus[]).map((s) => (
                  <option key={s} value={s}>{STATUS_META[s].label}</option>
                ))}
              </select>
            </label>
            <span className="ml-auto text-slate-500">共 {list.total} 筆</span>
          </div>
          {loading ? (
            <p className="py-8 text-center text-xs text-slate-500">載入中…</p>
          ) : list.items.length === 0 ? (
            <p className="py-8 text-center text-xs text-slate-500">尚無事件</p>
          ) : (
            <div className="grid gap-2 md:grid-cols-2">
              {list.items.map((ev) => (
                <AlertCard key={ev.id} event={ev} onOpen={setOpenId} />
              ))}
            </div>
          )}
          <div className="mt-3 flex items-center justify-center gap-2 text-xs">
            <button
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
              className="rounded bg-slate-200 px-2 py-1 disabled:opacity-40"
            >
              上一頁
            </button>
            <span className="tabular-nums text-slate-600">
              {page + 1} / {totalPages}
            </span>
            <button
              disabled={page + 1 >= totalPages}
              onClick={() => setPage((p) => p + 1)}
              className="rounded bg-slate-200 px-2 py-1 disabled:opacity-40"
            >
              下一頁
            </button>
          </div>
        </section>
      )}

      <EventDetailDrawer eventId={openId} onClose={() => setOpenId(null)} onUpdated={onUpdated} />
    </div>
  )
}
