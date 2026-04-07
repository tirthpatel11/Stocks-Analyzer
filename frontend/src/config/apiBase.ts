function defaultProdBase(): string {
  if (typeof window !== 'undefined' && window.location.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8001`;
  }
  return 'http://127.0.0.1:8001';
}

const explicitBase = (import.meta.env.VITE_API_URL as string | undefined)?.trim();

export const useDevProxy = import.meta.env.DEV && !explicitBase;

export function getAxiosBaseURL(): string {
  if (explicitBase) {
    return explicitBase.replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return '';
  }
  return defaultProdBase();
}

export function apiPath(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`;
  if (useDevProxy) {
    return `/api${p}`;
  }
  return p;
}

export function buildApiUrl(path: string): string {
  const p = apiPath(path);
  const base = getAxiosBaseURL();
  if (base === '') {
    return p;
  }
  return `${base.replace(/\/$/, '')}${p}`;
}
