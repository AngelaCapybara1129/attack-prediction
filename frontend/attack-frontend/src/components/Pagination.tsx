import { useState } from 'react'
import type { FormEvent } from 'react'
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'

type Props = {
  page: number // 0-based
  pageSize: number
  total: number
  onPageChange: (page: number) => void
  onPageSizeChange?: (size: number) => void
  pageSizeOptions?: number[]
  compact?: boolean // 只顯示頁碼列（放在列表上方）
}

/** 產生頁碼：1 … 4 5 [6] 7 8 … 20 */
function pageItems(current: number, totalPages: number): (number | 'gap')[] {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i)
  const items: (number | 'gap')[] = [0]
  const start = Math.max(1, current - 2)
  const end = Math.min(totalPages - 2, current + 2)
  if (start > 1) items.push('gap')
  for (let i = start; i <= end; i++) items.push(i)
  if (end < totalPages - 2) items.push('gap')
  items.push(totalPages - 1)
  return items
}

const btn =
  'inline-flex h-7 min-w-7 items-center justify-center rounded-md px-2 text-xs tabular-nums transition disabled:cursor-not-allowed disabled:opacity-35'

export default function Pagination({
  page,
  pageSize,
  total,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [10, 20, 30, 50],
  compact = false,
}: Props) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const [jump, setJump] = useState('')
  const go = (p: number) => onPageChange(Math.min(totalPages - 1, Math.max(0, p)))

  const onJump = (e: FormEvent) => {
    e.preventDefault()
    const n = parseInt(jump, 10)
    if (!Number.isNaN(n)) go(n - 1)
    setJump('')
  }

  const from = total === 0 ? 0 : page * pageSize + 1
  const to = Math.min(total, (page + 1) * pageSize)

  return (
    <nav aria-label="分頁" className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
      <span className="tabular-nums">
        第 {from}–{to} 筆，共 {total} 筆
      </span>

      <div className="flex items-center gap-1">
        <button className={`${btn} bg-white/70 hover:bg-white`} disabled={page === 0} onClick={() => go(0)} aria-label="第一頁">
          <ChevronsLeft className="h-3.5 w-3.5" />
        </button>
        <button className={`${btn} bg-white/70 hover:bg-white`} disabled={page === 0} onClick={() => go(page - 1)} aria-label="上一頁">
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
        {pageItems(page, totalPages).map((it, i) =>
          it === 'gap' ? (
            <span key={`gap${i}`} className="px-1 text-slate-400">…</span>
          ) : (
            <button
              key={it}
              onClick={() => go(it)}
              aria-current={it === page ? 'page' : undefined}
              className={`${btn} ${
                it === page ? 'bg-emerald-500 font-bold text-white' : 'bg-white/70 hover:bg-white'
              }`}
            >
              {it + 1}
            </button>
          ),
        )}
        <button
          className={`${btn} bg-white/70 hover:bg-white`}
          disabled={page + 1 >= totalPages}
          onClick={() => go(page + 1)}
          aria-label="下一頁"
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
        <button
          className={`${btn} bg-white/70 hover:bg-white`}
          disabled={page + 1 >= totalPages}
          onClick={() => go(totalPages - 1)}
          aria-label="最後一頁"
        >
          <ChevronsRight className="h-3.5 w-3.5" />
        </button>
      </div>

      {!compact && (
        <div className="flex items-center gap-2">
          {onPageSizeChange && (
            <label className="flex items-center gap-1">
              每頁
              <select
                value={pageSize}
                onChange={(e) => onPageSizeChange(Number(e.target.value))}
                className="rounded border border-slate-200 bg-white px-1 py-0.5"
              >
                {pageSizeOptions.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
              筆
            </label>
          )}
          {totalPages > 7 && (
            <form onSubmit={onJump} className="flex items-center gap-1">
              跳至
              <input
                value={jump}
                onChange={(e) => setJump(e.target.value.replace(/\D/g, ''))}
                inputMode="numeric"
                placeholder={`1–${totalPages}`}
                className="w-14 rounded border border-slate-200 bg-white px-1 py-0.5 text-center"
              />
              頁
            </form>
          )}
        </div>
      )}
    </nav>
  )
}
