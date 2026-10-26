import type { CleanupConsent } from "../../services/SmartCleanupService";

const key = (name: string) => `deep-reader.smart-cleanup:${encodeURIComponent(name)}`;

// Only consent locators are retained. Invalid storage is not permission to
// launch a fresh operation with an unknown predecessor.
export function recoverCleanup(name: string): CleanupConsent | null {
  const raw = sessionStorage.getItem(key(name));
  if (raw === null) return null;
  let value;
  try { value = JSON.parse(raw); } catch { throw new Error("cleanup_recovery_invalid"); }
  if (!value || typeof value.operation_id !== "string" || !/^[0-9a-f]{32}$/.test(value.operation_id)
      || typeof value.confirmation_token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(value.confirmation_token)) {
    throw new Error("cleanup_recovery_invalid");
  }
  return { operation_id: value.operation_id, confirmation_token: value.confirmation_token };
}
export function rememberCleanup(name: string, consent: CleanupConsent) {
  sessionStorage.setItem(key(name), JSON.stringify({ operation_id: consent.operation_id, confirmation_token: consent.confirmation_token }));
}
export function forgetCleanup(name: string, id: string) {
  if (recoverCleanup(name)?.operation_id === id) sessionStorage.removeItem(key(name));
}
