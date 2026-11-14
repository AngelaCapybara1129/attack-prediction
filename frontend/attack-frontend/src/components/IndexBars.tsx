import { INDEX_KEYS, INDEX_META } from '../constants/risk'
import type { IndexMap } from '../types/intent'

type IndexBarsProps = {
  values: IndexMap | null | undefined
  predicted?: IndexMap | null
  compact?: boolean
}

/** X1–X9 指數長條：灰線為表 2 門檻，藍色細線為 LSTM 預測的下一秒數值 */
export default function IndexBars({ values, predicted, compact = false }: IndexBarsProps) {
  return (
    <div className={compact ? 'space-y-1' : 'space-y-1.5'}>
      {INDEX_KEYS.map((key) => {
        const v = Math.max(0, Math.min(1, Number(values?.[key] ?? 0)))
        const p = predicted ? Math.max(0, Math.min(1, Number(predicted[key] ?? 0))) : null
        const meta = INDEX_META[key]
        const over = v >= meta.threshold
        return (
          <div key={key} className="flex items-center gap-2">
            <span className={`shrink-0 text-[10px] font-bold text-slate-600 ${compact ? 'w-14' : 'w-20'}`}>
              {key} {meta.name}
            </span>
            <div className="relative h-2 flex-1 overflow-visible rounded-full bg-slate-200">
              <div
                className={`h-2 rounded-full ${over ? 'bg-red-500' : v >= 0.3 ? 'bg-orange-400' : 'bg-slate-400'}`}
                style={{ width: `${v * 100}%` }}
              />
              <div
                className="absolute -top-0.5 h-3 w-px bg-slate-500/70"
                style={{ left: `${meta.threshold * 100}%` }}
                title={`門檻 ${meta.threshold}`}
              />
              {p !== null && (
                <div
                  className="absolute -top-1 h-4 w-0.5 rounded bg-blue-600"
                  style={{ left: `calc(${p * 100}% - 1px)` }}
                  title={`預測下一秒 ${p.toFixed(2)}`}
                />
              )}
            </div>
            <span className={`w-9 shrink-0 text-right text-[10px] tabular-nums ${over ? 'font-black text-red-600' : 'text-slate-500'}`}>
              {v.toFixed(2)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
