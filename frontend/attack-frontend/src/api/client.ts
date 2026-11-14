export const API_BASE: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export function getToken(): string {
  return localStorage.getItem('token') || ''
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : `HTTP ${res.status}`
    throw new ApiError(detail, res.status)
  }
  return data as T
}

/** <img src> 無法帶 Authorization header，改以 ?token= 傳遞 */
export function withToken(path: string): string {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}token=${encodeURIComponent(getToken())}`
}

export function wsUrl(path: string): string {
  const base = API_BASE.replace(/^http/, 'ws')
  return `${base}${path}?token=${encodeURIComponent(getToken())}`
}
