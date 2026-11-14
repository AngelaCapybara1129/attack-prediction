import { useEffect, useRef, useState } from 'react'
import { getToken, wsUrl } from '../api/client'
import type { SocketMessage } from '../types/intent'

/** 訂閱 /ws/alerts，斷線自動以指數退避重連 */
export function useAlertSocket(onMessage: (msg: SocketMessage) => void): { connected: boolean } {
  const [connected, setConnected] = useState(false)
  const handlerRef = useRef(onMessage)

  useEffect(() => {
    handlerRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry = 0
    let timer: number | undefined
    let disposed = false

    const connect = () => {
      if (disposed || !getToken()) return
      ws = new WebSocket(wsUrl('/ws/alerts'))
      ws.onopen = () => {
        retry = 0
        setConnected(true)
      }
      ws.onmessage = (e: MessageEvent<string>) => {
        try {
          handlerRef.current(JSON.parse(e.data) as SocketMessage)
        } catch {
          // 忽略無法解析的訊息
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (disposed) return
        retry += 1
        timer = window.setTimeout(connect, Math.min(15000, 1000 * 2 ** Math.min(retry, 4)))
      }
      ws.onerror = () => {
        ws?.close()
      }
    }

    connect()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
      ws?.close()
    }
  }, [])

  return { connected }
}
