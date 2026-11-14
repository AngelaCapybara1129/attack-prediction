import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { History, Search, X } from 'lucide-react'
import { apiFetch } from '../api/client'
import AlertCard from '../components/AlertCard'
import EventDetailDrawer from '../components/EventDetailDrawer'
import PageBackground from '../components/PageBackground'
import Pagination from '../components/Pagination'
import { INDEX_META, LEVELS, STATUS_META } from '../constants/risk'
import type { EventItem, EventListResponse, EventStatus, RiskLevel, TextSearchResponse } from '../types/intent'

const PAGE_SIZE = 10 // 每頁最多 10 筆事件
const SEARCH_LIMIT = 100 // 後端文字檢索上限；結果在前端分頁
const glassCard =
  'rounded-[16px] border border-white/50 bg-white/55 backdrop-blur-2xl shadow-[0_10px_30px_rgba(15,23,42,0.08)] ring-1 ring-white/60'

const EXAMPLES = ['上週穿黑衣揮拳', '深夜尾隨', '今天 L4', '持刀 cam0', '未處理 推擠']

/** 跨鏡頭軌跡追蹤與歷史證據回溯：事件列表 + 文字檢索 */
export default function EventsPage() {
  const [minLevel, setMinLevel] = useState<RiskLevel>('L1')
  const [status, setStatus] = useState<EventStatus | ''>('')
  const [page, setPage] = useState(0)
  const pageSize = PAGE_SIZE
  const [searchPage, setSearchPage] = useState(0)
  const topRef = useRef<HTMLDivElement>(null)
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
        limit: String(pageSize),
        offset: String(page * pageSize),
      })
      if (status) params.set('status', status)
      setList(await apiFetch<EventListResponse>(`/api/events?${params.toString()}`))
    } catch (err) {
      setError(err instanceof Error ? err.message : '讀取失敗')
    } finally {
      setLoading(false)
    }
  }, [minLevel, status, page, pageSize])

  useEffect(() => {
    void loadList()
  }, [loadList])

  // 資料變少（例如事件被刪除、篩選改變）時，避免停在不存在的頁
  useEffect(() => {
    const last = Math.max(0, Math.ceil(list.total / pageSize) - 1)
    if (!loading && page > last) setPage(last)
  }, [list.total, pageSize, page, loading])

  const scrollTop = () => topRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  const changePage = (p: number) => {
    setPage(p)
    scrollTop()
  }

  const changeSearchPage = (p: number) => {
    setSearchPage(p)
    scrollTop()
  }

  const runSearch = async (q: string) => {
    if (!q.trim()) {
      setSearch(null)
      return
    }
    setSearching(true)
    setSearchPage(0)
    setError('')
    try {
      setSearch(
        await apiFetch<TextSearchResponse>('/api/search/text', {
          method: 'POST',
          body: JSON.stringify({ query: q.trim(), limit: SEARCH_LIMIT }),
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

  const parsed = search?.parsed
  const searchItems = search ? search.items.slice(searchPage * pageSize, (searchPage + 1) * pageSize) : []

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden bg-white text-slate-900">
      <PageBackground />
      <div ref={topRef} className="relative z-10 mx-auto max-w-6xl scroll-mt-16 p-3 md:p-5">
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
            <span className="ml-auto">
              {search.items.length} 筆{search.items.length >= SEARCH_LIMIT ? `（僅顯示最相關的 ${SEARCH_LIMIT} 筆）` : ''}
            </span>
          </div>
          {search.items.length === 0 ? (
            <p className="py-8 text-center text-xs text-slate-500">沒有符合的事件</p>
          ) : (
            <>
            <div className="grid gap-2 md:grid-cols-2">
              {searchItems.map((it) => (
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
            <div className="mt-3">
              <Pagination
                page={searchPage}
                pageSize={pageSize}
                total={search.items.length}
                onPageChange={changeSearchPage}
              />
            </div>
            </>
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
          <div className="mt-3">
            <Pagination
              page={page}
              pageSize={pageSize}
              total={list.total}
              onPageChange={changePage}
            />
          </div>
        </section>
      )}

      <EventDetailDrawer eventId={openId} onClose={() => setOpenId(null)} onUpdated={onUpdated} />
      </div>
    </div>
  )
}
