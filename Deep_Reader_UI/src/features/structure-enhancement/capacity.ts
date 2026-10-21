import type { EnhancementCapacity } from "../../types/api";

export function sourceReadinessMessage(reason?: string | null): string {
  switch (reason) {
    case "extraction_identity_unknown":
      return "This document's saved source needs verification before smart structuring. Open Library to prepare it again.";
    case "source_changed":
      return "The source file has changed. Open Library to prepare the current file before smart structuring.";
    case "source_missing":
      return "The source file is missing. Restore it in Library before smart structuring.";
    case "source_ambiguous":
      return "More than one source file matches this document. Resolve the duplicate files before smart structuring.";
    default:
      return "Accepted source is unavailable. Open Library to prepare this document explicitly.";
  }
}

export function capacityMessage(capacity?: EnhancementCapacity | null): string | null {
  switch (capacity?.status) {
    case "fits":
      return capacity.full_source === true ? null
        : "Complete document text could not be confirmed. Smart structuring is not ready.";
    case "exceeded":
      return "This document is too long for smart structuring.";
    case "unavailable":
      return "Smart structuring is unavailable because the service is not configured.";
    case "resource_limited":
      return "This document exceeds the service's processing limits.";
    default:
      return "A service update is required before smart structuring can run.";
  }
}

// Only the documented pre-dispatch error contract proves no generation occurred.
export function isCapacityRejection(status: number, detail?: string, capacity?: EnhancementCapacity | null) {
  return Boolean(capacity && (
    (status === 413 && capacity.status === "exceeded" && detail === "model_capacity_exceeded") ||
    (status === 413 && capacity.status === "resource_limited" && detail === "local_resource_limit") ||
    (status === 503 && capacity.status === "unavailable" && detail === "capacity_unavailable")
  ));
}
