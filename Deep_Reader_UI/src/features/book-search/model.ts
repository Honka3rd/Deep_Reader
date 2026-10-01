import type { DocumentListItem } from "../../types/api";

export function toDocumentOptions(documents: DocumentListItem[]): string[] {
  return documents.map((item) => item.doc_name);
}

export function isSelectedDocumentOption(value: string, options: string[]): boolean {
  return options.includes(value.trim());
}
